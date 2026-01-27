import torch
import torch.nn as nn
from torchtyping import TensorType
from Self_Attention_Mechanism import SingleHeadAttention

class MultiHeadedSelfAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        self.attention_heads = nn.ModuleList([SingleHeadAttention(embedding_dim=embedding_dim, attention_dim=attention_dim//num_heads) for i in range(num_heads)])
        self.out_proj = nn.Linear(attention_dim, attention_dim, bias=True)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        output_list = []
        for head in self.attention_heads:
            output_list.append(head(embedded))
        out = torch.cat(output_list, dim=-1)       # (B,T,D)
        return self.out_proj(out)                 