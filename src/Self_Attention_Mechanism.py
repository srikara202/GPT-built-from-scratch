import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        self.key = nn.Linear(embedding_dim, attention_dim,bias=True)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=True)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=True)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key(embedded)
        q = self.query(embedded)
        v = self.value(embedded)
        kt = torch.transpose(k, 1, 2)
        qkt = torch.matmul(q,kt).div_(k.shape[-1]**0.5)
        # pre_mask = torch.tril(torch.ones(qkt.shape[-2], qkt.shape[-1], device=qkt.device))
        # qkt.masked_fill_(pre_mask==0,float('-inf'))
        # sm = self.softmax(qkt)
        T = qkt.size(-1)
        mask = torch.tril(torch.ones(T, T, device=qkt.device)).bool()
        qkt = qkt.masked_fill(~mask, float('-inf'))
        sm = torch.softmax(qkt, dim=-1)
        smv = torch.matmul(sm,v)
        return smv