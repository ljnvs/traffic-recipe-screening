"""Execute locked source rules; default fixed/nearest execution needs no fits."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import source_rules as core


def run(methods,pools,check_locked=True,recalibrate=False):
    data=json.loads((ROOT/'results/source_data.json').read_text(encoding='utf-8'))
    locked=json.loads((ROOT/'configs/locked_source_rules.json').read_text(encoding='utf-8'))
    cities=sorted({core.city(data,t) for t in data['labels']})
    targets=sorted(set(data['tasks'])-set(data['labels']))
    assert len(cities)==8 and len(targets)==8 and not set(targets)&set(data['early'])
    memo={}; results=[]
    for pool_name in pools:
        pool=sorted(c for c in data['descriptors'] if pool_name=='full23' or c.startswith('components_'))
        def fetch(train,spec,seed):
            key=core.digest(dict(cities=train,pool=pool,spec=spec,seed=seed))
            if key not in memo:
                bundle=core.fit_estimator(data,train,pool,spec,seed)
                memo[key]=core.predict_estimator(bundle,data,sorted(data['tasks']))
            return memo[key]
        engine=core.FlowCalibrationEngine(data,pool,fetch)
        for frozen in locked:
            if frozen['pool']!=pool_name or frozen['method'] not in methods:
                continue
            config=frozen['config']
            if recalibrate:
                config,_=engine.tune(cities,core.final_grids()[frozen['method']],frozen['meta_seed'])
            selected,baseline=engine.choose(cities,targets,config,frozen['meta_seed'])
            choices=dict(zip(targets,selected))
            if check_locked:
                assert choices==frozen['recommendations'],(frozen['method'],choices)
                assert baseline==frozen['source_baseline']
            results.append(dict(pool=pool_name,method=frozen['method'],meta_seed=frozen['meta_seed'],
                                recommendations=choices,matches_locked=choices==frozen['recommendations']))
    return dict(results=results,unique_estimator_fits=len(memo),target_labels_read=False,
                recalibrated=recalibrate)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',default=['S00_fixed','S02_nearest'])
    parser.add_argument('--pools',nargs='+',choices=['full23','neural17'],default=['full23','neural17'])
    parser.add_argument('--recalibrate',action='store_true',help='Run source-only finite calibration; requires sklearn for learned rules')
    parser.add_argument('--no-check-locked',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run(args.methods,args.pools,not args.no_check_locked,args.recalibrate)
    text=json.dumps(result,indent=2)
    if args.output: args.output.write_text(text+'\n',encoding='utf-8')
    print(json.dumps(dict(rule_records=len(result['results']),unique_estimator_fits=result['unique_estimator_fits'],
                          all_match=all(r['matches_locked'] for r in result['results']),target_labels_read=False)))
