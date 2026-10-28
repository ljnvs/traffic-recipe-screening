"""Representative configuration -> fit -> validation choice -> test scoring.

demo and dry-run never train. train is an explicit, potentially long action.
This portable entry point does not resume or replace the original experiments.
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from representative_panel import Panel,synthetic_arrays,no_fit_checks
from screen_candidates import screen


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def train(panel,args,*,runtime_smoke=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from l1_stable_execution_v92 import StableNativePredictor
    from native_city_model_v37 import native_config,initialize_named,backward_masked_batch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    policy=json.loads((ROOT/'configs/representative_protocol.json').read_text(encoding='utf-8'))
    if runtime_smoke:
        # The dedicated synthetic checker opts in; normal train retains the
        # complete published schedule and unchanged configuration file.
        policy.update(rounds=2,eligible_round=1,origins_per_round=6,logical_batch=2)
    bank={r['recipe_id']:r for r in json.loads((ROOT/'configs/recipes.json').read_text(encoding='utf-8'))}
    feasible=[r['recipe_id'] for r in screen()['feasible']]
    selected_ids=feasible if args.recipes==['all'] else args.recipes
    if not selected_ids or len(selected_ids)!=len(set(selected_ids)) or any(c not in feasible for c in selected_ids):
        raise ValueError('Select distinct feasible recipes; see screen_candidates.py')
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',dict(policy=policy,recipes=selected_ids,seed=args.seed,
                                    horizon_minutes=panel.horizon*15,rate_factor=panel.rate_factor,
                                    representative_portable_run=True,runtime_verification_only=runtime_smoke))
    allowed,valid=panel.origins('train'),panel.origins('validation')
    if len(allowed)<policy['origins_per_round']:
        raise ValueError('Insufficient training origins')

    def make(candidate):
        torch.manual_seed(args.seed)
        if args.device=='cuda':torch.cuda.manual_seed_all(args.seed)
        cfg=native_config(bank[candidate],panel.task)
        model=StableNativePredictor(cfg,panel.nodes,panel.horizon,96)
        initialize_named(model,args.seed)
        return model.to(args.device)

    def predict(model,origins):
        model.eval();values=[]
        with torch.inference_mode():
            for origin in origins:
                x=panel.inference_inputs(np.array([origin]),model.lookback)
                pred=model(*[torch.as_tensor(x[k],device=args.device) for k in ('x','slots','context')])
                if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite prediction')
                values.append(pred.cpu().numpy())
        return panel.native_unscale(np.concatenate(values))[1]

    candidates={}
    for candidate in selected_ids:
        model=make(candidate)
        optimizer=torch.optim.Adam(model.parameters(),lr=policy['lr'],weight_decay=policy['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,policy['rounds'],eta_min=policy['eta_min'])
        teacher=torch.Generator(device=args.device).manual_seed(args.seed+2718)
        shuffle=np.random.default_rng(args.seed)
        best,best_epoch,best_state=None,0,None
        updates=0;curve=[]
        for epoch in range(1,policy['rounds']+1):
            chosen=allowed[shuffle.permutation(len(allowed))[:policy['origins_per_round']]]
            model.train()
            for begin in range(0,len(chosen),policy['logical_batch']):
                ids=chosen[begin:begin+policy['logical_batch']]
                optimizer.zero_grad(set_to_none=True)
                prob=200/(200+math.exp(min(updates/200,700))) if model.cfg['decoder']=='autoregressive' else 0.
                result=backward_masked_batch(model,panel.gather(ids,model.lookback),1,args.device,prob,teacher)
                if result['empty']:continue
                torch.nn.utils.clip_grad_norm_(model.parameters(),policy['clip_norm'],error_if_nonfinite=True)
                optimizer.step();updates+=1
            error=panel.score(predict(model,valid),valid)['mae']
            curve.append(dict(round=epoch,validation_mae=error,updates=updates,lr=optimizer.param_groups[0]['lr']))
            if epoch>=policy['eligible_round'] and (best is None or error<best-1e-10):
                best,best_epoch=error,epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            scheduler.step()
        if best_state is None:raise RuntimeError('No eligible checkpoint')
        torch.save(best_state,output/(candidate+'_checkpoint.pt'))
        write(output/(candidate+'_curve.json'),curve)
        candidates[candidate]=dict(validation_mae=best,best_round=best_epoch,updates=updates)
        del model
        print(candidate,'validation MAE',best,flush=True)
    simple={}
    if args.include_simple:
        from simple_candidates import fit_validation
        simple=fit_validation(panel)
        for name,item in simple.items():
            candidates[name]={k:v for k,v in item.items() if k!='state'}
            if item['state'] is not None:np.savez_compressed(output/(name+'_checkpoint.npz'),**item['state'])
    # This selection expression can only see validation errors.
    chosen=min(candidates,key=lambda c:(candidates[c]['validation_mae'],c))
    test_ids=panel.origins('test')
    if chosen in simple:
        from simple_candidates import predictions
        pred=predictions(panel,test_ids,chosen,simple[chosen]['state'])
    else:
        model=make(chosen)
        model.load_state_dict(torch.load(output/(chosen+'_checkpoint.pt'),map_location='cpu',weights_only=True))
        pred=predict(model,test_ids)
    np.savez_compressed(output/'selected_test_predictions.npz',origins=test_ids,prediction_native=pred)
    result=dict(selected=chosen,candidate_validation=candidates,test=panel.score(pred,test_ids),
                test_labels_used_for_selection=False,representative_portable_run=True,
                runtime_verification_only=runtime_smoke)
    write(output/'result.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['demo','dry-run','train'])
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--horizon-minutes',type=int,choices=[15,60],default=15)
    parser.add_argument('--rate-factor',type=float,choices=[1.,4.],default=1.)
    parser.add_argument('--seed',type=int,default=20261006)
    parser.add_argument('--recipes',nargs='+',default=['004','005'])
    parser.add_argument('--include-simple',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.mode=='demo':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise FileExistsError(args.output)
        np.savez_compressed(args.output,**synthetic_arrays())
        print('Synthetic format example:',args.output)
    else:
        if args.input is None:parser.error('--input is required')
        panel=Panel.load(args.input,horizon_minutes=args.horizon_minutes,rate_factor=args.rate_factor)
        if args.mode=='dry-run':
            result=no_fit_checks(panel)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            write(args.output,result);print(json.dumps(result))
        else:train(panel,args)
