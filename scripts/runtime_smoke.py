"""Minimal entry-point verification; outputs are not paper experiments.

predictor runs the real portable training loop on a three-node synthetic
panel, two rounds and six training origins. source replays locked learned
rules on existing source labels and checks one excluded-region estimator.
The published protocol, score exports and locked choices are never edited.
"""
from pathlib import Path
from argparse import ArgumentParser,Namespace
from importlib import metadata
import hashlib,json,platform,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import representative_task as task
from representative_panel import Panel,synthetic_arrays,no_fit_checks

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def versions():
    result={'python':platform.python_version()}
    for name in ('numpy','torch','scikit-learn','scipy','joblib'):
        try:result[name]=metadata.version(name)
        except metadata.PackageNotFoundError:result[name]=None
    return result
def frozen():
    return {p.relative_to(ROOT).as_posix():sha(p) for folder in ('results','configs') for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}

class SmokePanel(Panel):
    def origins(self,split):
        full=super().origins(split)
        indices=[0,1,28,31,50,60] if split=='train' else [0,1,9,29]
        return full[indices]

def predictor(output):
    import torch
    from l1_stable_execution_v92 import StableNativePredictor
    from native_city_model_v37 import native_config,initialize_named
    bank={r['recipe_id']:r for r in json.loads((ROOT/'configs/recipes.json').read_text(encoding='utf-8'))}
    before=frozen();checks=[];all_parameter_updates=0
    for horizon,rate in ((15,1.),(60,4.)):
        arrays=synthetic_arrays()
        no_fit_checks(Panel(arrays,horizon_minutes=horizon,rate_factor=rate))
        panel=SmokePanel(arrays,horizon_minutes=horizon,rate_factor=rate)
        args=Namespace(output=output/f'predictor_{horizon}',recipes=['004','005'],
                       seed=20261006,device='cpu',include_simple=False)
        result=task.train(panel,args,runtime_smoke=True)
        assert result['runtime_verification_only'] and not result['test_labels_used_for_selection']
        assert result['selected']==min(result['candidate_validation'],key=lambda c:(result['candidate_validation'][c]['validation_mae'],c))
        loaded={};updates={}
        for candidate,record in result['candidate_validation'].items():
            assert record['updates']==6 and record['best_round'] in (1,2)
            torch.manual_seed(args.seed)
            model=StableNativePredictor(native_config(bank[candidate],panel.task),panel.nodes,panel.horizon,96)
            initialize_named(model,args.seed)
            initial={k:v.detach().clone() for k,v in model.named_parameters()}
            state=torch.load(args.output/(candidate+'_checkpoint.pt'),map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            assert all(torch.isfinite(v).all() for v in state.values())
            changed=sum(not torch.equal(initial[k],v) for k,v in model.named_parameters())
            assert changed>0
            updates[candidate]=changed;all_parameter_updates+=changed
            loaded[candidate]=model.eval()
        model=loaded[result['selected']];origins=panel.origins('test');values=[]
        with torch.inference_mode():
            for origin in origins:
                x=panel.inference_inputs(np.array([origin]),model.lookback)
                pred=model(*[torch.as_tensor(x[k]) for k in ('x','slots','context')])
                values.append(pred.numpy())
        pred=panel.native_unscale(np.concatenate(values))[1]
        with np.load(args.output/'selected_test_predictions.npz',allow_pickle=False) as saved:
            assert np.array_equal(saved['origins'],origins)
            np.testing.assert_allclose(saved['prediction_native'],pred,rtol=0,atol=0)
        assert panel.score(pred,origins)==result['test']
        checks.append(dict(horizon_minutes=horizon,rate_factor=rate,candidates=['004','005'],
            rounds=2,train_origins=6,validation_origins=4,test_origins=4,updates_per_candidate=6,
            finite_gradients_and_checkpoints=True,parameters_changed_by_optimisation=updates,
            validation_only_selection_verified=True,checkpoint_reload_matches_predictions_exactly=True,
            native_unit_test_rescoring_matches=True))
    assert before==frozen()
    summary=dict(passed=True,verification_only=True,synthetic_predictor_data=True,
        training_runtime_executed=True,predictor_fits=4,scientific_experiments_rerun=False,
        published_protocol_executed=False,published_results_unchanged=True,versions=versions(),checks=checks)
    write(output/'summary.json',summary)
    return summary

def source(output):
    import replay_source_rules as replay
    import source_rules as core
    before=frozen()
    result=replay.run(['S07_FFORMS','S09'],['full23'])
    assert len(result['results'])==6 and result['unique_estimator_fits']==30
    assert all(r['matches_locked'] for r in result['results'])
    assert not result['target_labels_read'] and not result['recalibrated']
    write(output/'locked_replay.json',result)
    data=core.read(ROOT/'results/source_data.json')
    cities=sorted({core.city(data,t) for t in data['labels']})
    pool=sorted(data['descriptors'])
    counts={};held_city=None
    for held in cities:
        train=[c for c in cities if c!=held]
        encoder=core.TaskEncoder(data,train)
        expected={t for t in data['labels'] if core.city(data,t)!=held}
        assert set(encoder.tids)==expected
        assert not (set(encoder.tids)&set(core.task_ids(data,[held])))
        counts[held]=len(encoder.tids)
        if len(encoder.tids)==16 and held_city is None:held_city=held
    assert sorted(counts.values())==[16,16,18,18,18,18,18,18]
    spec=next(r['config']['estimator'] for r in core.read(ROOT/'configs/locked_source_rules.json')
              if r['method']=='S09' and r['pool']=='full23' and r['meta_seed']==20261010)
    train=[c for c in cities if c!=held_city]
    bundle=core.fit_estimator(data,train,pool,spec,20261010)
    assert len(bundle['source_tasks'])==16 and set(bundle['encoder'].tids)==set(bundle['source_tasks'])
    assert all(core.city(data,t)!=held_city for t in bundle['source_tasks'])
    held_tasks=core.task_ids(data,[held_city])
    predicted=core.predict_estimator(bundle,data,held_tasks)
    assert np.isfinite(predicted).all()
    assert before==frozen()
    summary=dict(passed=True,verification_only=True,learned_source_runtime_executed=True,
        versions=versions(),methods=['S07_FFORMS','S09'],pool='full23',meta_seeds=core.SEEDS,
        locked_rule_records=6,target_choices_checked=48,all_locked_choices_match=True,
        learned_estimator_fits=31,locked_replay_estimator_fits=30,outer_fold_estimator_checks=1,
        all_outer_preprocessing_task_counts=counts,held_region_used_for_fitting=False,
        target_validation_or_test_labels_read=False,full_nested_calibration_executed=False,
        source_scores_and_locked_configs_unchanged=True,scientific_experiments_rerun=False)
    write(output/'summary.json',summary)
    return summary

if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['predictor','source'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result={'predictor':predictor,'source':source}[args.mode](args.output)
    print(json.dumps(result,indent=2))
