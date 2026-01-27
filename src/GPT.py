import torch
import torch.nn as nn
from torchtyping import TensorType
from Transformer_Block import TransformerBlock

class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.context_length = context_length
        self.vocab_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(context_length, model_dim)
        self.nxTransformer = nn.Sequential()
        for i in range(num_blocks):
            self.nxTransformer.append(TransformerBlock(model_dim, num_heads))
        self.norm = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, vocab_size)
        # self.softmax = nn.Softmax(dim=-1)


    def forward(self, context: TensorType[int]) -> TensorType[float]:
        pos = torch.arange(context.shape[-1], device=context.device).unsqueeze(0)
        input_embedding = torch.add(self.vocab_embedding(context), self.pos_embedding(pos))
        nxtrans = self.nxTransformer(input_embedding)
        normalized = self.norm(nxtrans)
        logits = self.linear(normalized)
        # smax = self.softmax(lin)
        return logits