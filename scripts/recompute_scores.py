"""Recompute paired gains, four-region decisions, outer summaries and intervals."""
from pathlib import Path
from statistics import mean,stdev
import argparse
import csv
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[20261006,20261007,20261008]


def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def short(name):
    return name.rsplit('_',1)[-1] if name.startswith('components_') else name.removesuffix('_external')


def recompute():
    idx={}
    with (ROOT/'results/four_region_scores.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            for seed in SEEDS if r['shared_across_forecast_seeds']=='True' else [int(r['forecast_seed'])]:
                idx[(r['city'],int(r['horizon_minutes']),seed,r['candidate'])]=dict(val=float(r['validation_mae']),test=float(r['test_mae']))
    tasks=sorted({k[:3] for k in idx})
    locked=load('configs/locked_source_rules.json')
    regions={'Madrid':'T01','Innsbruck':'T02','Greater Geelong':'T03','Latrobe':'T04'}
    choices={}
    for r in locked:
        if r['pool']=='full23' and r['meta_seed']==20261010 and r['method'].startswith('S'):
            choices[r['method']]={t:short(r['recommendations'][f'{regions[t[0]]}:flow:{t[1]}']) for t in tasks}
    for pool in ('neural17','full23'):
        choices[pool]={}
        for t in tasks:
            eligible=[k[-1] for k in idx if k[:3]==t and (k[-1].isdigit() or (pool=='full23' and k[-1] not in ('GraphWaveNet','STAEformer')))]
            choices[pool][t]=min(eligible,key=lambda c:(idx[(*t,c)]['val'],c))
    for c in ('004','005'):choices['fixed'+c]={t:c for t in tasks}
    summaries={}
    for scope in ('native3','all4','derived1'):
        keep=[t for t in tasks if scope=='all4' or (t[0]=='Innsbruck')==(scope=='derived1')]
        summaries[scope]={m:{'gain_vs'+ref:mean(100*(1-idx[(*t,p[t])]['test']/idx[(*t,ref)]['test']) for t in keep)
                             for ref in ('004','005')} for m,p in choices.items()}
    four=[]
    order=['Madrid','Greater Geelong','Latrobe','Innsbruck']
    for city,h in sorted({t[:2] for t in tasks},key=lambda t:(order.index(t[0]),t[1])):
        tt=[t for t in tasks if t[:2]==(city,h)]
        columns={}
        for method in ('fixed004','fixed005','neural17','full23','S02_nearest','S07_FFORMS','S09'):
            vals=[idx[(*t,choices[method][t])]['test'] for t in tt]
            columns[method]=dict(choices=[choices[method][t] for t in tt],mean=mean(vals),sd=stdev(vals))
        for model in ('Persistence','GraphWaveNet','STAEformer'):
            vals=[idx[(*t,model)]['test'] for t in tt]
            columns[model]=dict(mean=mean(vals),sd=stdev(vals))
        four.append(dict(city=city,horizon=h,scope='derived' if city=='Innsbruck' else 'native',columns=columns))
    outer=[]
    raw=load('results/source_outer_holdouts.json')
    for method in ['S00_fixed','S01_majority','S02_nearest','S03_oldGB','S04_direct','S05_matched','S07_FFORMS','S08_ID','S09',
                   'A_generic_description','A_generic_ID','A_no_H','A_no_u','A_no_gate']:
        rr=[r['flow_only'] for r in raw if r['pool']=='full23' and r['method']==method]
        assert all(abs(mean(r['city_gain_pct'].values())-r['macro_gain_pct'])<1e-12 for r in rr)
        outer.append(dict(method=method,mean_gain=mean(r['macro_gain_pct'] for r in rr),
                          worst_loss=max(r['worst_city_loss_pct'] for r in rr),
                          switch_min=min(r['switch_count'] for r in rr),switch_max=max(r['switch_count'] for r in rr)))
    daily={}
    for r in load('results/daily_error_sums.json'):
        key=(r['city'],r['horizon_minutes'],r['forecast_seed'],r['candidate'])
        daily.setdefault(key,dict(sums=np.zeros(7),counts=np.zeros(7)))
        daily[key]['sums'][r['day_index']]=r['absolute_error_sum']
        daily[key]['counts'][r['day_index']]=r['valid_targets']
    intervals=[]
    for length in (1,2):
        rng=np.random.default_rng(20261022+length);draws={}
        for provider in ('Madrid','Victoria'):
            starts=rng.integers(0,8-length,size=(10000,int(np.ceil(7/length))))
            draws[provider]=(starts[...,None]+np.arange(length)[None,None,:]).reshape(10000,-1)[:,:7]
        for method in ('neural17','S02_nearest'):
            for ref in ('005','004'):
                values=[]
                for t in [t for t in tasks if t[0]!='Innsbruck']:
                    draw=draws['Madrid' if t[0]=='Madrid' else 'Victoria']
                    def error(c):
                        x=daily[(*t,c)]
                        assert abs(x['sums'].sum()/x['counts'].sum()-idx[(*t,c)]['test'])<1e-10
                        return x['sums'][draw].sum(1)/x['counts'][draw].sum(1)
                    values.append(100*(1-error(choices[method][t])/error(ref)))
                lo,hi=np.percentile(np.mean(values,axis=0),[2.5,97.5])
                intervals.append(dict(method=method,reference=ref,days=length,lo=float(lo),hi=float(hi)))
    costs={}
    with (ROOT/'results/native_three_resource_scores.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for method in ('fixed004','fixed005','neural17','S02_nearest'):
        candidates=[r for r in rows if r['forecast_seed'] and short(r['model_id']).isdigit()]
        if method!='neural17':
            candidates=[r for r in candidates if short(r['model_id'])==choices[method][(r['city'],int(r['horizon_min']),int(r['forecast_seed']))]]
        costs[method]=dict(target_neural_fits=len(candidates),workflow_minutes=sum(float(r['workflow_seconds']) for r in candidates)/60)
    return dict(four_region=four,summaries=summaries,source_outer=outer,time_intervals=intervals,costs=costs)


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        for key,value in expected.items():compare(actual[key],value,path+'/'+key)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+f'/{i}')
    elif isinstance(expected,(int,float)):
        assert abs(actual-expected)<1e-10,(path,actual,expected)
    else:assert actual==expected,(path,actual,expected)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=recompute()
    expected=load('results/expected_summaries.json')
    for name in ('four_region','summaries','source_outer','time_intervals'):compare(result[name],expected[name],name)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(expected_summaries_match=True,source_outer_rules=len(result['source_outer']),
                          time_intervals=len(result['time_intervals']),costs=result['costs'],new_fits=0)))
