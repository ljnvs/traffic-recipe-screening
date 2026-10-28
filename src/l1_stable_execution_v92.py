"""Shape-stable FP32 execution for new L1 panels, with unchanged parameters.

Each origin follows the existing NativePredictor batch-of-one computation.
Outer microbatch size only groups these identical calls. Loss normalization
and accumulation remain in backward_masked_batch. This trades throughput for
a fixed numerical path; no candidate, activation, precision or loss changes.
AR teacher choices remain shared over a logical batch exactly as in v37.
"""
import torch
from native_city_model_v37 import NativePredictor


class StableNativePredictor(NativePredictor):
    def forward(self, history, daily_slots=None, periodic_history=None,
                teacher=None, teacher_probability=0., generator=None, teacher_mask=None):
        if history.shape[0] < 1:
            raise ValueError('Empty model batch')
        sampling = self.training and self.cfg['decoder']=='autoregressive' and teacher_probability>0 and self.horizon>1
        if sampling and generator is None:
            raise ValueError('Explicit teacher generator required')
        initial = generator.get_state().clone() if sampling else None
        advanced = None; result = []
        def row(value, i): return value[i:i+1] if value is not None else None
        for i in range(history.shape[0]):
            if sampling: generator.set_state(initial)
            result.append(super().forward(history[i:i+1], row(daily_slots,i), row(periodic_history,i),
                row(teacher,i), teacher_probability, generator, row(teacher_mask,i)))
            if sampling and advanced is None: advanced=generator.get_state().clone()
        if sampling: generator.set_state(advanced)
        return torch.cat(result,dim=0)
