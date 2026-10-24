"""Correct full-forecast autoregressive feedback; direct paths delegate to v6.

The historical v6 predictor stays untouched. Parameter names, initialization,
graph construction and every direct decoder operation are unchanged.
"""
import torch
from component_predictor_v6 import TrafficComponentPredictor as _HistoricalPredictor
from component_predictor_v6 import initialize_named, parameter_hash


class TrafficComponentPredictor(_HistoricalPredictor):
    def forward(self, history, daily_slots=None, periodic_history=None, teacher=None,
                teacher_probability=0., generator=None):
        if self.cfg["decoder"] != "autoregressive":
            return super().forward(history, daily_slots, periodic_history, teacher, teacher_probability, generator)
        if history.ndim == 3:
            history = history.unsqueeze(-1)
        if tuple(history.shape[1:]) != (self.lookback, self.nodes, 1):
            raise ValueError("Expected [B,L,N,1] numeric target history")
        if not self.training and (teacher is not None or teacher_probability != 0):
            raise ValueError("Inference cannot consume future labels")
        if teacher is not None and teacher.shape != (history.shape[0], self.horizon, self.nodes):
            raise ValueError("Teacher labels have wrong horizon/node layout")
        if teacher_probability > 0 and teacher is None:
            raise ValueError("Teacher sampling requires autoregressive training labels")
        x, context = history, periodic_history
        if self.periodic_context and (context is None or context.shape != (x.shape[0], self.nodes, 2*self.horizon)):
            raise ValueError("Observed daily/week history missing or wrong shape")
        instance = self.cfg["normalization"] == "instance_scale"
        if instance:
            center = x.mean(1, keepdim=True).detach()
            scale = (x.var(1, keepdim=True, unbiased=False)+1e-5).sqrt().detach()
            x = (x-center)/scale
            if context is not None:
                context = (context-center[:,0,:,0,None])/scale[:,0,:,0,None]
            if teacher is not None:
                teacher = (teacher-center[...,0])/scale[...,0]
        periodic = self.period_head(context).transpose(1,2) if self.periodic_context else None
        latent = self._encode(x, daily_slots)
        supports = self.graph_bank()
        pool = getattr(self.graph_bank,"pool_embedding",None)
        state = torch.zeros(x.shape[0],self.nodes,32,device=x.device,dtype=x.dtype)
        encoder_parameters = self.encoder_cell.prepare(pool)
        for t in range(self.lookback):
            state = self.encoder_cell(latent[:,t],state,supports,pool,encoder_parameters)
        previous = torch.zeros(x.shape[0],self.nodes,1,device=x.device,dtype=x.dtype)
        decoder_parameters = self.decoder_cell.prepare(pool)
        predictions = []
        for t in range(self.horizon):
            state = self.decoder_cell(previous,state,supports,pool,decoder_parameters)
            complete = self.step_head(state)
            if periodic is not None:
                complete = complete + periodic[:,t,:,None]
            predictions.append(complete[...,0])
            # Both alternatives are the full target in the same normalized
            # coordinates. The common periodic branch is added exactly once.
            previous = complete
            if self.training and teacher_probability > 0 and t < self.horizon-1:
                choose = torch.rand((),device=x.device,generator=generator) < teacher_probability
                previous = torch.where(choose,teacher[:,t,:,None],complete)
        output = torch.stack(predictions,1)
        if instance:
            output = output*scale[...,0]+center[...,0]
        return output
