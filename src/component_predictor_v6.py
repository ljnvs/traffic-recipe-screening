"""Executable canonical traffic recipes from the v5 mechanism design.

This implements standardized mechanism combinations, not faithful reproduction
of the seven whole paper models. Daily/week history is an explicit common
control. Future labels are permitted only as prior decoder inputs in training.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from component_constraints_v5 import violations
from traffic_component_primitives_v5 import AxisAttention, polynomial_features

CURRENT_CAPS = {"provided_graph":True,"symmetric_graph":True,"relative_daily_slot":True,
                "verified_calendar":False,"training_semantic_graph":False,
                "training_pattern_prototypes":False,"laplacian_positions":False}


def fixed_graph_supports(raw):
    """Binary benchmark graph -> declared walk transpose and symmetric spectrum."""
    a = np.asarray(raw,dtype=np.float64).copy()
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all() or (a<0).any():
        raise ValueError("Adjacency must be a finite nonnegative square matrix")
    np.fill_diagonal(a,0.)
    degree = a.sum(1)
    walk = a / np.where(degree>0,degree,1)[:,None]
    empty = np.flatnonzero(degree==0)
    walk[empty,empty]=1.
    # Symmetrization is an explicit construction, not an assumption about
    # directed traffic flow. Current audited CSVs use binary undirected edges.
    symmetric = np.maximum(a,a.T)
    ds=symmetric.sum(1)
    inv=np.zeros_like(ds); inv[ds>0]=1/np.sqrt(ds[ds>0])
    lap=np.eye(len(a))-inv[:,None]*symmetric*inv[None,:]
    lap[np.flatnonzero(ds==0),np.flatnonzero(ds==0)]=0.
    largest=float(np.linalg.eigvalsh(lap).max())
    scaled=2*lap/largest-np.eye(len(a)) if largest>1e-12 else -np.eye(len(a))
    return walk.T.astype(np.float32), scaled.astype(np.float32)


class GraphBank(nn.Module):
    def __init__(self,nodes,source,parameterization,raw,embedding_dim=8):
        super().__init__()
        walk,lap=fixed_graph_supports(raw)
        self.source=source
        self.register_buffer("raw_adjacency",torch.tensor(np.asarray(raw,dtype=np.float32)))
        self.register_buffer("walk_target_source",torch.tensor(walk))
        self.register_buffer("scaled_laplacian",torch.tensor(lap))
        self.support_count=2 if source=="provided_plus_adaptive" else (0 if source=="none" else 1)
        if source in ("adaptive_asymmetric","provided_plus_adaptive"):
            self.source_embedding=nn.Parameter(torch.empty(nodes,embedding_dim))
            self.target_embedding=nn.Parameter(torch.empty(embedding_dim,nodes))
        if source=="adaptive_tied" or parameterization=="node_pool":
            self.pool_embedding=nn.Parameter(torch.empty(nodes,embedding_dim))

    def forward(self):
        if self.source=="none": return []
        if self.source=="provided_walk": return [self.walk_target_source]
        if self.source=="provided_laplacian": return [self.scaled_laplacian]
        if self.source=="adaptive_tied":
            return [F.softmax(F.relu(self.pool_embedding@self.pool_embedding.T),-1)]
        adaptive=F.softmax(F.relu(self.source_embedding@self.target_embedding),-1).T
        return [adaptive] if self.source=="adaptive_asymmetric" else [self.walk_target_source,adaptive]


class GraphFeatureLayer(nn.Module):
    def __init__(self,input_dim,output_dim,cfg,support_count,order=2):
        super().__init__()
        self.cfg=dict(cfg); self.order=order if cfg["spatial_operator"]!="none" else 0
        self.support_count=support_count
        self.basis="powers" if cfg["spatial_operator"]=="powers_graph_conv" else "chebyshev_reset"
        self.layout="basis_major"
        if (cfg["assembly"]=="graph_recurrent" and cfg["spatial_operator"]=="chebyshev_graph_conv"
                and cfg["graph_source"] in ("provided_walk","provided_plus_adaptive") and cfg["parameterization"]=="shared"):
            self.basis,self.layout="dcrnn_tf_legacy","feature_major"
        terms=1+support_count*self.order
        self.input_dim,self.output_dim=input_dim,output_dim
        if cfg["parameterization"]=="node_pool":
            self.weight_pool=nn.Parameter(torch.empty(8,terms,input_dim,output_dim))
            self.bias_pool=nn.Parameter(torch.empty(8,output_dim))
        else:
            self.linear=nn.Linear(terms*input_dim,output_dim)

    def prepare(self,pool_embedding):
        if self.cfg["parameterization"]=="shared": return None
        if pool_embedding is None: raise ValueError("NAPL node embedding missing")
        return (torch.einsum("ne,ekio->nkio",pool_embedding,self.weight_pool),pool_embedding@self.bias_pool)

    def forward(self,x,supports,pool_embedding=None,prepared=None):
        features=polynomial_features(x,supports,self.order,self.basis,"target_source",self.layout)
        if self.cfg["parameterization"]=="shared": return self.linear(features)
        if pool_embedding is None: raise ValueError("NAPL node embedding missing")
        weights,bias=self.prepare(pool_embedding) if prepared is None else prepared
        expanded=features.reshape(*features.shape[:-1],-1,self.input_dim)
        return torch.einsum("btnki,nkio->btno",expanded,weights)+bias


class EfficientTemporal(nn.Module):
    """Temporal convolution by explicit patches, avoiding legacy Conv1d bottleneck."""
    def __init__(self,width,kernel=3,dilation=1,mode="glu",causal=False):
        super().__init__()
        self.width,self.kernel,self.dilation,self.mode,self.causal=width,kernel,dilation,mode,causal
        self.linear=nn.Linear(kernel*width,(2 if mode in ("glu","gated") else 1)*width)

    def forward(self,x):
        span=1+(self.kernel-1)*self.dilation
        if self.causal: padded=F.pad(x,(0,0,0,0,span-1,0)); length=x.shape[1]
        else: padded=x; length=x.shape[1]-span+1
        if length<1: raise ValueError("Temporal kernel exceeds available history")
        patch=torch.stack([padded[:,k*self.dilation:k*self.dilation+length] for k in range(self.kernel)],-2).flatten(-2)
        value=self.linear(patch)
        if self.mode=="glu":
            left,right=value.chunk(2,-1)
            return (left+x[:,-length:])*torch.sigmoid(right)
        if self.mode=="gated":
            left,right=value.chunk(2,-1)
            return torch.tanh(left)*torch.sigmoid(right)
        return F.relu(value+x[:,-length:])


class SandwichBlock(nn.Module):
    def __init__(self,width,cfg,support_count):
        super().__init__()
        self.first=EfficientTemporal(width,3,mode="glu")
        self.spatial=GraphFeatureLayer(width,width,cfg,support_count)
        self.last=EfficientTemporal(width,3,mode="relu")
        self.norm=nn.LayerNorm(width)

    def forward(self,x,supports,pool_embedding):
        return self.norm(self.last(F.relu(self.spatial(self.first(x),supports,pool_embedding))))


class DilatedGraphBlock(nn.Module):
    def __init__(self,width,cfg,support_count,dilation):
        super().__init__()
        self.temporal=EfficientTemporal(width,2,dilation,mode="gated",causal=True)
        self.spatial=GraphFeatureLayer(width,width,cfg,support_count)
        self.norm=nn.LayerNorm(width)

    def forward(self,x,supports,pool_embedding):
        temporal=self.temporal(x)
        return self.norm(x+self.spatial(temporal,supports,pool_embedding)),temporal


class GraphGRUCell(nn.Module):
    def __init__(self,input_dim,hidden_dim,cfg,support_count):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.gate=GraphFeatureLayer(input_dim+hidden_dim,2*hidden_dim,cfg,support_count)
        self.candidate=GraphFeatureLayer(input_dim+hidden_dim,hidden_dim,cfg,support_count)

    def prepare(self,pool_embedding):
        return self.gate.prepare(pool_embedding),self.candidate.prepare(pool_embedding)

    def forward(self,x,state,supports,pool_embedding,prepared=None):
        prepared=self.prepare(pool_embedding) if prepared is None else prepared
        merged=torch.cat((x,state),-1).unsqueeze(1)
        reset,update=torch.sigmoid(self.gate(merged,supports,pool_embedding,prepared[0])[:,0]).chunk(2,-1)
        proposal=torch.tanh(self.candidate(torch.cat((x,reset*state),-1).unsqueeze(1),supports,pool_embedding,prepared[1])[:,0])
        return update*state+(1-update)*proposal


class AttentionBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.attention=AxisAttention(width,4)
        self.norm1,self.norm2=nn.LayerNorm(width),nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width))

    def forward(self,x,axis):
        x=self.norm1(x+self.attention(x,axis))
        return self.norm2(x+self.ff(x))


class TrafficComponentPredictor(nn.Module):
    def __init__(self,cfg,nodes,horizon,raw_adjacency,periodic_context=True):
        super().__init__()
        bad=violations(cfg,CURRENT_CAPS)
        if bad: raise ValueError("; ".join(bad))
        self.cfg,self.nodes,self.horizon=dict(cfg),nodes,horizon
        self.lookback=int(cfg["lookback"]); self.periodic_context=periodic_context
        self.graph_bank=GraphBank(nodes,cfg["graph_source"],cfg["parameterization"],raw_adjacency)
        self.base_width=32
        if cfg["node_encoding"]=="node_identity":
            self.node_identity=nn.Parameter(torch.empty(nodes,8))
        elif cfg["node_encoding"]=="adaptive_slot":
            self.adaptive_slot=nn.Parameter(torch.empty(self.lookback,nodes,8))
        if cfg["calendar"]=="relative_daily_slot": self.day_embedding=nn.Embedding(288,8)
        extra=8*int(cfg["node_encoding"]!="none")+8*int(cfg["calendar"]!="none")
        self.width=32+extra
        assembly=cfg["assembly"]
        if assembly=="node_mlp":
            if cfg["node_encoding"]=="adaptive_slot": self.value_projection=nn.Linear(1,32)
            else: self.history_projection=nn.Linear(self.lookback,32)
            self.mlp_blocks=nn.ModuleList([nn.Sequential(nn.Linear(self.width,self.width),nn.ReLU(),nn.Linear(self.width,self.width)) for _ in range(2)])
            self.head=nn.Linear(self.lookback*self.width if cfg["node_encoding"]=="adaptive_slot" else self.width,horizon)
        else:
            self.value_projection=nn.Linear(1,32)
            if assembly=="graph_sandwich":
                if self.lookback-8<=1: raise ValueError("Two valid ST blocks leave an invalid output window")
                self.blocks=nn.ModuleList([SandwichBlock(self.width,cfg,self.graph_bank.support_count) for _ in range(2)])
                self.head=nn.Linear((self.lookback-8)*self.width,horizon)
            elif assembly=="dilated_graph_stack":
                # Same number of layers for both lookbacks; all observed slots feed
                # the direct head, so lookback changes do not silently change depth.
                self.blocks=nn.ModuleList([DilatedGraphBlock(self.width,cfg,self.graph_bank.support_count,d) for d in (1,2,4,8)])
                self.head=nn.Linear(self.lookback*self.width,horizon)
            elif assembly=="axis_attention":
                self.temporal_attention=nn.ModuleList([AttentionBlock(self.width) for _ in range(2)])
                self.node_attention=nn.ModuleList([AttentionBlock(self.width) for _ in range(2)]) if cfg["spatial_operator"]=="dense_attention" else nn.ModuleList()
                self.head=nn.Linear(self.lookback*self.width,horizon)
            elif assembly=="graph_recurrent":
                self.encoder_cell=GraphGRUCell(self.width,32,cfg,self.graph_bank.support_count)
                if cfg["decoder"]=="direct": self.head=nn.Linear(32,horizon)
                else:
                    self.decoder_cell=GraphGRUCell(1,32,cfg,self.graph_bank.support_count)
                    self.step_head=nn.Linear(32,1)
            else: raise ValueError(assembly)
        if periodic_context:
            self.period_head=nn.Linear(2*horizon,horizon)
            nn.init.zeros_(self.period_head.weight); nn.init.zeros_(self.period_head.bias)

    def _encode(self,x,daily_slots):
        if self.cfg["calendar"]!="none":
            if daily_slots is None or daily_slots.shape!=(x.shape[0],self.lookback): raise ValueError("Daily slot history missing")
            if bool(((daily_slots<0)|(daily_slots>=288)).any()): raise ValueError("Daily slots out of range")
        assembly=self.cfg["assembly"]
        if assembly=="node_mlp" and self.cfg["node_encoding"]!="adaptive_slot":
            value=self.history_projection(x[...,0].transpose(1,2)); extras=[]
            if self.cfg["node_encoding"]=="node_identity": extras.append(self.node_identity.unsqueeze(0).expand(x.shape[0],-1,-1))
            if self.cfg["calendar"]!="none": extras.append(self.day_embedding(daily_slots[:,-1]).unsqueeze(1).expand(-1,self.nodes,-1))
            return torch.cat((value,*extras),-1)
        parts=[self.value_projection(x)]
        if self.cfg["node_encoding"]=="node_identity": parts.append(self.node_identity[None,None].expand(x.shape[0],self.lookback,-1,-1))
        if self.cfg["node_encoding"]=="adaptive_slot": parts.append(self.adaptive_slot.unsqueeze(0).expand(x.shape[0],-1,-1,-1))
        if self.cfg["calendar"]!="none": parts.append(self.day_embedding(daily_slots).unsqueeze(2).expand(-1,-1,self.nodes,-1))
        return torch.cat(parts,-1)

    def forward(self,history,daily_slots=None,periodic_history=None,teacher=None,teacher_probability=0.,generator=None):
        if history.ndim==3: history=history.unsqueeze(-1)
        if tuple(history.shape[1:])!=(self.lookback,self.nodes,1): raise ValueError("Expected [B,L,N,1] numeric target history")
        if not self.training and (teacher is not None or teacher_probability!=0): raise ValueError("Inference cannot consume future labels")
        if teacher is not None and teacher.shape!=(history.shape[0],self.horizon,self.nodes): raise ValueError("Teacher labels have wrong horizon/node layout")
        if teacher_probability>0 and (teacher is None or self.cfg["decoder"]!="autoregressive"): raise ValueError("Teacher sampling requires autoregressive training labels")
        x=history; context=periodic_history
        if self.periodic_context and (context is None or context.shape!=(x.shape[0],self.nodes,2*self.horizon)):
            raise ValueError("Observed daily/week history missing or wrong shape")
        if self.cfg["normalization"]=="instance_scale":
            center=x.mean(1,keepdim=True).detach()
            scale=(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt().detach()
            x=(x-center)/scale
            if context is not None: context=(context-center[:,0,:,0,None])/scale[:,0,:,0,None]
            if teacher is not None: teacher=(teacher-center[...,0])/scale[...,0]
        latent=self._encode(x,daily_slots)
        supports=self.graph_bank()
        pool=getattr(self.graph_bank,"pool_embedding",None)
        assembly=self.cfg["assembly"]
        if assembly=="node_mlp":
            for block in self.mlp_blocks: latent=latent+block(latent)
            summary=latent.transpose(1,2).flatten(-2) if latent.ndim==4 else latent
            output=self.head(summary).transpose(1,2)
        elif assembly in ("graph_sandwich","dilated_graph_stack"):
            if assembly=="graph_sandwich":
                for block in self.blocks: latent=block(latent,supports,pool)
            else:
                skip=0
                for block in self.blocks:
                    latent,part=block(latent,supports,pool); skip=skip+part
                latent=latent+skip
            output=self.head(latent.transpose(1,2).flatten(-2)).transpose(1,2)
        elif assembly=="axis_attention":
            for block in self.temporal_attention: latent=block(latent,"time")
            for block in self.node_attention: latent=block(latent,"nodes")
            output=self.head(latent.transpose(1,2).flatten(-2)).transpose(1,2)
        else:
            state=torch.zeros(x.shape[0],self.nodes,32,device=x.device,dtype=x.dtype)
            encoder_parameters=self.encoder_cell.prepare(pool)
            for t in range(self.lookback): state=self.encoder_cell(latent[:,t],state,supports,pool,encoder_parameters)
            if self.cfg["decoder"]=="direct": output=self.head(state).transpose(1,2)
            else:
                previous=torch.zeros(x.shape[0],self.nodes,1,device=x.device,dtype=x.dtype); predictions=[]
                decoder_parameters=self.decoder_cell.prepare(pool)
                for t in range(self.horizon):
                    state=self.decoder_cell(previous,state,supports,pool,decoder_parameters)
                    predicted=self.step_head(state); predictions.append(predicted[...,0])
                    previous=predicted
                    # Only the preceding target can become the next step's input.
                    if self.training and teacher_probability>0 and t<self.horizon-1:
                        choose=torch.rand((),device=x.device,generator=generator)<teacher_probability
                        previous=torch.where(choose,teacher[:,t,:,None],predicted)
                output=torch.stack(predictions,1)
        if self.periodic_context: output=output+self.period_head(context).transpose(1,2)
        if self.cfg["normalization"]=="instance_scale": output=output*scale[...,0]+center[...,0]
        return output


def initialize_named(model,seed):
    """Same name/shape -> same FP32 values, without copying graph buffers."""
    layernorm_weights={id(m.weight) for m in model.modules() if isinstance(m,nn.LayerNorm)}
    with torch.no_grad():
        for name,p in model.named_parameters():
            code=f"{seed}|{name}|{tuple(p.shape)}"
            rng=torch.Generator(device="cpu").manual_seed(int(hashlib.sha256(code.encode()).hexdigest()[:15],16))
            if name.startswith("period_head."): p.zero_()
            elif id(p) in layernorm_weights: p.fill_(1)
            elif p.ndim==1 or name.endswith("bias_pool"): p.zero_()
            elif "embedding" in name or "adaptive_slot" in name or "node_identity" in name: p.normal_(0,.1,generator=rng)
            else: nn.init.xavier_uniform_(p,generator=rng)


def parameter_hash(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode()); digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
