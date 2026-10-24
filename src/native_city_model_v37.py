"""Keep canonical mechanisms; native slots and valid-only AR teacher feedback."""
import torch
from torch import nn
from component_predictor_v7b import TrafficComponentPredictor as PreviousPredictor
from component_predictor_v6 import initialize_named,parameter_hash
from component_constraints_v5 import violations

CAPS={'provided_graph':False,'symmetric_graph':False,'relative_daily_slot':True,'verified_calendar':False,'training_semantic_graph':False,'training_pattern_prototypes':False,'laplacian_positions':False}


def native_config(original,task):
    config=dict(original);seconds=int(config['lookback'])*300;step=int(task['frequency_seconds'])
    if seconds%step:raise ValueError('Nonintegral physical history duration')
    config['lookback']=seconds//step
    problems=violations(config,CAPS)
    if config['assembly']=='graph_sandwich' and config['lookback']-8<=1:problems.append('Original valid ST blocks do not fit native physical history')
    if problems:raise ValueError('; '.join(problems))
    return config


class NativePredictor(PreviousPredictor):
    def __init__(self,config,nodes,horizon,slots_per_day):
        # This shape-only identity is never exposed as an available road graph.
        import numpy as np
        if violations(config,CAPS):raise ValueError('Unavailable graph/capability combination')
        super().__init__(config,nodes,horizon,np.eye(nodes,dtype=np.float32))
        self.slots_per_day=int(slots_per_day)
        if config['calendar']=='relative_daily_slot' and self.slots_per_day!=288:self.day_embedding=nn.Embedding(self.slots_per_day,8)

    def _encode(self,x,daily_slots):
        if self.cfg['calendar']!='none':
            if daily_slots is None or daily_slots.shape!=(x.shape[0],self.lookback):raise ValueError('Native daily slot history missing')
            if bool(((daily_slots<0)|(daily_slots>=self.slots_per_day)).any()):raise ValueError('Native daily slots out of range')
        assembly=self.cfg['assembly']
        if assembly=='node_mlp' and self.cfg['node_encoding']!='adaptive_slot':
            value=self.history_projection(x[...,0].transpose(1,2));extras=[]
            if self.cfg['node_encoding']=='node_identity':extras.append(self.node_identity.unsqueeze(0).expand(x.shape[0],-1,-1))
            if self.cfg['calendar']!='none':extras.append(self.day_embedding(daily_slots[:,-1]).unsqueeze(1).expand(-1,self.nodes,-1))
            return torch.cat((value,*extras),-1)
        parts=[self.value_projection(x)]
        if self.cfg['node_encoding']=='node_identity':parts.append(self.node_identity[None,None].expand(x.shape[0],self.lookback,-1,-1))
        if self.cfg['node_encoding']=='adaptive_slot':parts.append(self.adaptive_slot.unsqueeze(0).expand(x.shape[0],-1,-1,-1))
        if self.cfg['calendar']!='none':parts.append(self.day_embedding(daily_slots).unsqueeze(2).expand(-1,-1,self.nodes,-1))
        return torch.cat(parts,-1)

    def forward(self,history,daily_slots=None,periodic_history=None,teacher=None,teacher_probability=0.,generator=None,teacher_mask=None):
        if not self.training and teacher_mask is not None:raise ValueError('Inference must not receive future quality masks')
        if self.cfg['decoder']!='autoregressive':
            if teacher_mask is not None:raise ValueError('Teacher mask only belongs to AR training')
            return super().forward(history,daily_slots,periodic_history,teacher,teacher_probability,generator)
        if history.ndim==3:history=history.unsqueeze(-1)
        if tuple(history.shape[1:])!=(self.lookback,self.nodes,1):raise ValueError('Wrong native history shape')
        if not self.training and (teacher is not None or teacher_probability!=0):raise ValueError('Inference cannot consume future targets')
        if teacher_probability>0 and teacher is None:raise ValueError('AR teacher sampling needs training targets')
        if teacher is not None and teacher.shape!=(history.shape[0],self.horizon,self.nodes):raise ValueError('Wrong trueH teacher shape')
        if teacher_mask is not None and (teacher is None or teacher_mask.shape!=teacher.shape or teacher_mask.dtype!=torch.bool):raise ValueError('Teacher mask must align with training target')
        x,context=history,periodic_history
        if self.periodic_context and (context is None or context.shape!=(x.shape[0],self.nodes,2*self.horizon)):raise ValueError('Past day/week segments missing')
        instance=self.cfg['normalization']=='instance_scale'
        if instance:
            center=x.mean(1,keepdim=True).detach();scale=(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt().detach();x=(x-center)/scale
            if context is not None:context=(context-center[:,0,:,0,None])/scale[:,0,:,0,None]
            if teacher is not None:teacher=(teacher-center[...,0])/scale[...,0]
        periodic=self.period_head(context).transpose(1,2) if self.periodic_context else None;latent=self._encode(x,daily_slots);supports=self.graph_bank();pool=getattr(self.graph_bank,'pool_embedding',None)
        state=torch.zeros(x.shape[0],self.nodes,32,device=x.device,dtype=x.dtype);prepared=self.encoder_cell.prepare(pool)
        for t in range(self.lookback):state=self.encoder_cell(latent[:,t],state,supports,pool,prepared)
        previous=torch.zeros(x.shape[0],self.nodes,1,device=x.device,dtype=x.dtype);prepared=self.decoder_cell.prepare(pool);predictions=[]
        for t in range(self.horizon):
            state=self.decoder_cell(previous,state,supports,pool,prepared);complete=self.step_head(state)
            if periodic is not None:complete=complete+periodic[:,t,:,None]
            predictions.append(complete[...,0]);previous=complete
            if self.training and teacher_probability>0 and t<self.horizon-1:
                choose=torch.rand((),device=x.device,generator=generator)<teacher_probability
                allowed=choose if teacher_mask is None else choose&teacher_mask[:,t,:,None]
                previous=torch.where(allowed,teacher[:,t,:,None],complete)
        output=torch.stack(predictions,1)
        if instance:output=output*scale[...,0]+center[...,0]
        return output


def loss_elements(pred,target,name):
    if name=='mae':return (pred-target).abs()
    if name=='mse':return (pred-target).square()
    if name=='huber':return torch.nn.functional.huber_loss(pred,target,reduction='none',delta=1.)
    raise ValueError(name)


def backward_masked_batch(model,data,microbatch,device,teacher_probability=0.,generator=None,collect=False):
    count=int(data['mask'].sum())
    if count==0:return {'empty':True,'loss':None,'predictions':None,'valid_targets':0}
    autoregressive=model.cfg['decoder']=='autoregressive';sampling=autoregressive and teacher_probability>0 and model.horizon>1
    if sampling and generator is None:raise ValueError('Logical sampling generator missing')
    initial=generator.get_state().clone() if sampling else None;advanced=None;total=0.;parts=[]
    for start in range(0,len(data['x']),microbatch):
        sl=slice(start,start+microbatch);args=[torch.as_tensor(data[k][sl],device=device) for k in ('x','slots','context')];target=torch.as_tensor(data['y'][sl],device=device);mask=torch.as_tensor(data['mask'][sl],device=device)
        if sampling:generator.set_state(initial)
        output=model(*args,teacher=target,teacher_probability=teacher_probability,generator=generator,teacher_mask=mask) if autoregressive else model(*args)
        if sampling and advanced is None:advanced=generator.get_state().clone()
        element=loss_elements(output,target,model.cfg['loss']);loss=element.masked_select(mask).sum()/count
        if not torch.isfinite(loss) or not torch.isfinite(output).all():raise RuntimeError('Nonfinite masked logical batch')
        loss.backward();total+=float(loss.detach())
        if collect:parts.append(output.detach().cpu())
    if sampling:generator.set_state(advanced)
    return {'empty':False,'loss':total,'predictions':torch.cat(parts) if collect else None,'valid_targets':count}
