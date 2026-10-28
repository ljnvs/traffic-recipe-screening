"""Six simple candidates with the original shared-node Ridge definition."""
import numpy as np
from simple_core import native_features,weighted_moments,ridge_solve


def predictions(panel,origins,name,state=None):
    if name=='Persistence':
        raw=np.repeat(panel.filled[origins-1,None,:],panel.horizon,axis=1)/panel.rate_factor
    elif name=='DayWeekAverage':
        q=origins[:,None]+np.arange(panel.horizon)
        raw=(panel.filled[q-96]+panel.filled[q-672])/(2*panel.rate_factor)
    else:
        length,calendar=int(name.split('_')[1][1:]),name.split('_')[2]
        x=native_features(panel,origins,length,calendar)[0]
        z=(((x-state['feature_mean'])/state['feature_scale'])@state['coefficient'].T+state['intercept'])
        z=z.reshape(len(origins),panel.nodes,panel.horizon).transpose(0,2,1)
        return panel.native_unscale(z)[1]
    return np.maximum(raw.astype(np.float32),0)


def fit_validation(panel):
    origins={s:panel.origins(s) for s in ('train','validation')}
    result={}
    for length in (12,36):
        for calendar in ('none','daily'):
            name=f'Ridge_l{length}_{calendar}'
            # The original functions expect gather fields y/mask only; they
            # never consume test observations or test-derived scalers.
            data={s:native_features(panel,ids,length,calendar) for s,ids in origins.items()}
            center=data['train'][0].mean(0)
            scale=data['train'][0].std(0)
            scale=np.where(scale>1e-12,scale,1.)
            x=(data['train'][0]-center)/scale
            moments=weighted_moments(x,data['train'][1],data['train'][2])
            states,errors={},{}
            for alpha in (.1,1.,10.,100.):
                coef,intercept=ridge_solve(moments,alpha)
                state=dict(coefficient=coef,intercept=intercept,feature_mean=center,feature_scale=scale)
                states[alpha]=state
                errors[alpha]=panel.score(predictions(panel,origins['validation'],name,state),origins['validation'])['mae']
            alpha=min(errors,key=lambda a:(errors[a],a))
            result[name]=dict(validation_mae=errors[alpha],alpha=alpha,state=states[alpha])
    for name in ('Persistence','DayWeekAverage'):
        result[name]=dict(validation_mae=panel.score(predictions(panel,origins['validation'],name),origins['validation'])['mae'],state=None)
    return result
