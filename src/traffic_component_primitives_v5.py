"""Auditable component ports. These are not seven reproduced forecasting models.

Canonical numerical tensors: [B,T,N,F]. Graph multiplication direction, basis
layout and legacy support recurrence are explicit rather than silently unified.
"""
import torch
from torch import nn
from torch.nn import functional as F


def graph_message(x, support, orientation="target_source"):
    if orientation == "target_source":
        return torch.einsum("ij,btjf->btif", support, x)
    if orientation == "source_target":
        return torch.einsum("btif,ij->btjf", x, support)
    raise ValueError(orientation)


def polynomial_features(x, supports, order, basis="powers", orientation="target_source", layout="basis_major"):
    if x.ndim != 4 or order < 0:
        raise ValueError("Expected [B,T,N,F] and nonnegative order")
    blocks = [x]
    legacy_previous = x
    for support in supports:
        if tuple(support.shape) != (x.shape[2], x.shape[2]):
            raise ValueError("Support/node shape mismatch")
        if order == 0: continue
        previous = legacy_previous if basis == "dcrnn_tf_legacy" else x
        current = graph_message(previous, support, orientation)
        blocks.append(current)
        for _ in range(2, order+1):
            if basis == "powers":
                newer = graph_message(current, support, orientation)
            elif basis in ("chebyshev_reset", "dcrnn_tf_legacy"):
                newer = 2 * graph_message(current, support, orientation) - previous
            else:
                raise ValueError(basis)
            blocks.append(newer)
            previous, current = current, newer
        if basis == "dcrnn_tf_legacy":
            legacy_previous = previous
    if layout == "basis_major": return torch.cat(blocks, dim=-1)
    if layout == "feature_major":
        return torch.stack(blocks, dim=-1).flatten(-2)
    raise ValueError(layout)


class PolynomialGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, support_count=1, order=2, basis="powers", orientation="source_target", layout="basis_major"):
        super().__init__()
        self.support_count, self.order = support_count, order
        self.basis, self.orientation, self.layout = basis, orientation, layout
        self.linear = nn.Linear(input_dim*(1+support_count*order), output_dim)

    def forward(self, x, supports):
        if len(supports) != self.support_count: raise ValueError("Support count changed after parameter initialization")
        return self.linear(polynomial_features(x, supports, self.order, self.basis, self.orientation, self.layout))


class AsymmetricAdaptiveGraph(nn.Module):
    def __init__(self, nodes, embedding_dim=8):
        super().__init__()
        self.source_embedding = nn.Parameter(torch.randn(nodes, embedding_dim))
        self.target_embedding = nn.Parameter(torch.randn(embedding_dim, nodes))

    def forward(self):
        return F.softmax(F.relu(self.source_embedding @ self.target_embedding), dim=-1)


class TiedAdaptiveGraphConv(nn.Module):
    """AGCRN AVWGCN port, retaining the shared E for DAGG and NAPL."""
    def __init__(self, input_dim, output_dim, nodes, embedding_dim=8, cheb_k=3):
        super().__init__()
        if cheb_k < 2: raise ValueError("This source-compatible port requires cheb_k>=2")
        self.nodes, self.cheb_k = nodes, cheb_k
        self.embedding = nn.Parameter(torch.randn(nodes, embedding_dim))
        self.weight_pool = nn.Parameter(torch.empty(embedding_dim, cheb_k, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.empty(embedding_dim, output_dim))
        nn.init.xavier_uniform_(self.weight_pool); nn.init.zeros_(self.bias_pool)

    def graph(self):
        return F.softmax(F.relu(self.embedding @ self.embedding.T), dim=-1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] != self.nodes: raise ValueError("Expected [B,T,N,F] for the fixed node table")
        support = self.graph()
        terms = [torch.eye(self.nodes, device=x.device, dtype=x.dtype), support]
        for _ in range(2, self.cheb_k): terms.append(2*support @ terms[-1] - terms[-2])
        basis = torch.stack(terms)
        node_weights = torch.einsum("nd,dkio->nkio", self.embedding, self.weight_pool)
        node_bias = self.embedding @ self.bias_pool
        graph_features = torch.einsum("knm,btmc->btnkc", basis, x)
        return torch.einsum("btnki,nkio->btno", graph_features, node_weights) + node_bias


class IdentityHistoryEncoder(nn.Module):
    """STID's history projection + static node identity + residual per-node MLP.

    Calendar embeddings and final forecast head are outside this prototype.
    """
    def __init__(self, nodes, lookback, input_dim=1, value_dim=16, node_dim=8, layers=2):
        super().__init__()
        self.nodes, self.lookback, self.input_dim = nodes, lookback, input_dim
        self.history_projection = nn.Linear(lookback*input_dim, value_dim)
        self.node_embedding = nn.Parameter(torch.empty(nodes, node_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        width = value_dim+node_dim
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width,width), nn.ReLU(), nn.Linear(width,width)) for _ in range(layers)])

    def forward(self, x):
        if tuple(x.shape[1:]) != (self.lookback,self.nodes,self.input_dim): raise ValueError("History contract mismatch")
        values = self.history_projection(x.transpose(1,2).flatten(-2))
        identity = self.node_embedding.unsqueeze(0).expand(x.shape[0],-1,-1)
        z = torch.cat((values,identity),dim=-1)
        for block in self.blocks: z = z + block(z)
        return z


class AxisAttention(nn.Module):
    """STAEformer-style self attention on time or node axis, with source-shaped QKV."""
    def __init__(self, width=16, heads=4):
        super().__init__()
        if width % heads: raise ValueError("Width must be divisible by heads")
        self.width, self.heads = width, heads
        self.q, self.k, self.v, self.output = (nn.Linear(width,width) for _ in range(4))

    def forward(self, x, axis="nodes", blocked_mask=None):
        if axis == "time":
            sequence = x.permute(0,2,1,3)
        elif axis == "nodes": sequence = x
        else: raise ValueError(axis)
        b, group, length, width = sequence.shape
        def split(layer):
            return layer(sequence).reshape(b,group,length,self.heads,width//self.heads).transpose(2,3)
        q,k,v = split(self.q),split(self.k),split(self.v)
        logits = (q @ k.transpose(-2,-1)) / (width//self.heads)**.5
        if blocked_mask is not None:
            if blocked_mask.shape != (length,length) or bool(blocked_mask.all(-1).any()): raise ValueError("Mask must leave at least one key per row")
            logits = logits.masked_fill(blocked_mask, -torch.inf)
        weights = logits.softmax(-1)
        z = (weights @ v).transpose(2,3).reshape(b,group,length,width)
        out = self.output(z)
        return out.permute(0,2,1,3) if axis == "time" else out


def past_pattern_windows(x, size=3):
    if size < 1: raise ValueError(size)
    padded = F.pad(x, (0,0,0,0,size-1,0))
    return torch.stack([padded[:,i:i+x.shape[1]] for i in range(size)], dim=-2)


class DelayPatternKeyUpdate(nn.Module):
    """PDFormer pattern/key update after its pattern embedding, initially Fout=1."""
    def __init__(self, model_dim=16, geographic_dim=8, head_dim=2):
        super().__init__()
        self.q = nn.Linear(model_dim,geographic_dim)
        self.k = nn.Linear(model_dim,geographic_dim)
        self.v = nn.Linear(model_dim,geographic_dim)
        self.scale = head_dim**-.5

    def forward(self, geographic_keys, pattern_queries, prototype_embeddings):
        logits = self.q(pattern_queries) @ self.k(prototype_embeddings).T
        weights = (logits*self.scale).softmax(-1)
        return geographic_keys + weights @ self.v(prototype_embeddings)
