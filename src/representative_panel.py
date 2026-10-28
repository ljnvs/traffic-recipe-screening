"""Portable 30-day, 15-minute representative panel; no provider data included."""
from pathlib import Path
import numpy as np


def bounded_fill(raw, available, mean, cap_steps=4):
    """Same causal forward-fill as l1_runtime_v94; target quality is not used."""
    last = np.asarray(mean, np.float32).copy()
    last_tick = np.full(raw.shape[1], -1000000, np.int64)
    filled = np.empty(raw.shape, np.float32)
    for tick in range(len(raw)):
        ok = available[tick] & np.isfinite(raw[tick]) & (raw[tick] >= 0)
        last[ok] = raw[tick, ok]
        last_tick[ok] = tick
        filled[tick] = np.where(tick-last_tick <= cap_steps, last, mean)
    return filled


class Panel:
    """Input is native units; Victoria's internal hourly factor can be four."""
    slots_per_day, step, max_horizon = 96, 900, 4

    def __init__(self, arrays, horizon_minutes=15, rate_factor=1.):
        if horizon_minutes not in (15, 60) or rate_factor not in (1., 4.):
            raise ValueError('Use 15/60 minutes and native/hourly factor 1 or 4')
        self.native_raw = np.asarray(arrays['raw'], dtype=np.float32).copy()
        self.input_available = np.asarray(arrays['input_available']).copy()
        self.primary = np.asarray(arrays['target_valid']).copy()
        if self.native_raw.ndim != 2 or self.native_raw.shape[0] != 2880:
            raise ValueError('raw must have shape [2880, nodes] in provider order')
        if self.native_raw.shape[1] < 1:
            raise ValueError('At least one node is required')
        if any(x.shape != self.native_raw.shape or x.dtype != bool for x in (self.input_available, self.primary)):
            raise ValueError('Separate boolean input and target masks must match raw')
        if not (np.isfinite(self.native_raw[self.primary]) & (self.native_raw[self.primary] >= 0)).all():
            raise ValueError('Valid targets must be finite nonnegative observations')
        self.nodes = self.native_raw.shape[1]
        self.horizon, self.rate_factor = horizon_minutes//15, float(rate_factor)
        self.task = dict(frequency_seconds=900)
        means, scales = [], []
        for node in range(self.nodes):
            keep = self.input_available[:16*96, node] & np.isfinite(self.native_raw[:16*96,node]) & (self.native_raw[:16*96,node] >= 0)
            values = self.native_raw[:16*96,node][keep].astype(np.float64)
            if len(values) == 0:
                raise ValueError('Each node needs an input-available prefix observation')
            means.append(values.mean())
            sd = values.std(ddof=0)
            scales.append(sd if sd > 1e-8 else 1.)
        self.mean = np.asarray(means,np.float32)*self.rate_factor
        self.scale = np.asarray(scales,np.float32)*self.rate_factor
        self.raw = (self.native_raw*self.rate_factor).astype(np.float32)
        self.filled = bounded_fill(self.raw,self.input_available,self.mean,4)
        self.z = ((self.filled-self.mean)/self.scale).astype(np.float32)
        self.yz = ((np.where(self.primary,self.raw,self.mean)-self.mean)/self.scale).astype(np.float32)

    @classmethod
    def load(cls,path,**kwargs):
        with np.load(path,allow_pickle=False) as data:
            arrays = {key:data[key] for key in ('raw','input_available','target_valid')}
        return cls(arrays,**kwargs)

    def origins(self,split):
        begin,end = {'train':(7,16),'validation':(16,23),'test':(23,30)}[split]
        return np.arange(begin*96,end*96-self.max_horizon+1,dtype=np.int64)

    def lookback(self,original_steps):
        if original_steps*300 % self.step:
            raise ValueError('Nonintegral physical history')
        return original_steps*300//self.step

    def inference_inputs(self,origins,lookback):
        s = np.asarray(origins,dtype=np.int64)
        if s.ndim != 1 or not len(s) or lookback < 1:
            raise ValueError('Nonempty origin vector and positive lookback required')
        ids = s[:,None]+np.arange(-int(lookback),0)
        contexts = [s[:,None]-lag+np.arange(self.horizon) for lag in (96,672)]
        if ids.min() < 0 or ids.max() >= len(self.z) or not np.all(ids<s[:,None]):
            raise ValueError('Unavailable or future recent input')
        if any(q.min()<0 or q.max()>=len(self.z) or not np.all(q<s[:,None]) for q in contexts):
            raise ValueError('Unavailable or future periodic input')
        return dict(x=self.z[ids].copy(),slots=(ids%96).copy(),
                    context=np.concatenate([self.z[q].transpose(0,2,1) for q in contexts],axis=-1).copy())

    def gather(self,origins,lookback):
        data = self.inference_inputs(origins,lookback)
        s = np.asarray(origins,dtype=np.int64)
        q = s[:,None]+np.arange(self.horizon)
        if q.min()<0 or q.max()>=len(self.raw):
            raise ValueError('Target outside panel')
        return dict(data,origins=s,y=self.yz[q].copy(),mask=self.primary[q].copy())

    def native_unscale(self,z):
        raw = ((np.asarray(z,dtype=np.float64)*self.scale+self.mean)/self.rate_factor).astype(np.float32)
        return raw,np.maximum(raw,0)

    def score(self,pred,origins):
        q = np.asarray(origins)[:,None]+np.arange(self.horizon)
        mask,truth = self.primary[q],self.native_raw[q]
        if np.asarray(pred).shape != truth.shape or not np.isfinite(pred).all() or not mask.any():
            raise ValueError('Finite aligned predictions and valid targets required')
        error = np.asarray(pred)[mask].astype(np.float64)-truth[mask].astype(np.float64)
        return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.square(error).mean())),
                    valid_targets=int(mask.sum()),possible_targets=int(mask.size))


def synthetic_arrays():
    """Deterministic format example; observations do not represent a real city."""
    t = np.arange(2880)[:,None]
    raw = (30+5*np.sin(2*np.pi*t/96)+np.arange(3)[None,:]*7).astype(np.float32)
    available = np.ones(raw.shape,bool)
    valid = np.ones(raw.shape,bool)
    available[700:707,0] = False
    valid[700:707,0] = False
    raw[700:707,0] = np.nan
    valid[2304:2309,1] = False
    # An observed input need not be a valid target; keep the masks independent.
    return dict(raw=raw,input_available=available,target_valid=valid)


def no_fit_checks(panel):
    """Check split containment, causal inputs, masks and native-unit scoring."""
    for split,(begin,end) in {'train':(7,16),'validation':(16,23),'test':(23,30)}.items():
        origins = panel.origins(split)
        assert origins.min()>=begin*96 and origins.max()+4<=end*96
        inputs = panel.inference_inputs(origins[[0,-1]],12)
        assert set(inputs)=={'x','slots','context'}
        assert inputs['context'].shape==(2,panel.nodes,2*panel.horizon)
    origins = panel.origins('test')[:3]
    q = origins[:,None]+np.arange(panel.horizon)
    pred = panel.native_raw[q].astype(np.float64)
    pred[panel.primary[q]] += 2
    pred[~panel.primary[q]] = 1e6
    assert abs(panel.score(pred,origins)['mae']-2)<1e-7
    # Future masks and observations cannot enter input assembly after scaling
    # parameters have been frozen at the prefix boundary.
    from copy import deepcopy
    other = deepcopy(panel)
    origin = 24*96
    other.z[origin:] = 1e6
    other.primary[origin:] = ~other.primary[origin:]
    a,b = panel.inference_inputs(np.array([origin]),12),other.inference_inputs(np.array([origin]),12)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    return dict(passed=True,new_predictor_fits=0,new_selector_fits=0,
                train_origins=len(panel.origins('train')),validation_origins=len(panel.origins('validation')),
                test_origins=len(panel.origins('test')),horizon=panel.horizon)
