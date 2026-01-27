import torch
import torch.nn as nn
from torchtyping import TensorType
from Multi_Headed_Attention import MultiHeadedSelfAttention

class TransformerBlock(nn.Module):
    
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.mha = MultiHeadedSelfAttention(model_dim, model_dim, num_heads)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = self.FeedForward(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        x_norm = self.norm1(embedded)
        x_add = self.mha(x_norm) + embedded
        x_norm2 = self.norm2(x_add)
        x_ff_add = self.feed_forward(x_norm2) + x_add
        return x_ff_add
    
    class FeedForward(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.gelu = nn.GELU(approximate="tanh")
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2
        
        def forward(self, x: TensorType[float]) -> TensorType[float]:
            return self.dropout(self.down_projection(self.gelu(self.up_projection(x))))
