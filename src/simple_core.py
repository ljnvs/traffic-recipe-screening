"""Original NumPy feature and masked Ridge functions, extracted unchanged."""
import numpy as np

def weighted_moments(x,y,mask):
    items=[]
    for h in range(y.shape[1]):
        valid=mask[:,h];assert valid.any()
        xx=x[valid];yy=y[valid,h];mx=xx.mean(0);my=yy.mean();xc=xx-mx;yc=yy-my
        items.append((xc.T@xc,xc.T@yc,mx,my))
    return items

def ridge_solve(moments,alpha):
    coefficients=[];intercepts=[]
    for gram,cross,mx,my in moments:
        coefficient=np.linalg.solve(gram+alpha*np.eye(len(cross)),cross)
        coefficients.append(coefficient);intercepts.append(my-mx@coefficient)
    return np.asarray(coefficients),np.asarray(intercepts)

def native_features(panel,origins,original_lookback,calendar):
    data=panel.gather(origins,panel.lookback(original_lookback));x=data['x'].transpose(0,2,1);parts=[x,data['context']]
    if calendar=='daily':
        phase=2*np.pi*(origins%panel.slots_per_day)/panel.slots_per_day
        marks=np.stack((np.sin(phase),np.cos(phase)),axis=-1).astype(np.float32);parts.append(np.broadcast_to(marks[:,None],(len(origins),panel.nodes,2)))
    x=np.concatenate(parts,axis=-1);return x.reshape(-1,x.shape[-1]).astype(np.float64),data['y'].transpose(0,2,1).reshape(-1,panel.horizon).astype(np.float64),data['mask'].transpose(0,2,1).reshape(-1,panel.horizon)
