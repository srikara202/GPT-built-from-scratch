import torch
import tiktoken
from typing import Tuple, List
import yaml

def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        # Use safe_load to avoid arbitrary code execution
        config = yaml.safe_load(file)
    return config

# Load the configuration
config_data = load_hyperparameters('config.yaml')
vocab = config_data['vocab']


def batch_loader(raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[int]]]:
    torch.manual_seed(0)
    X = []
    Y = []
    enc = tiktoken.get_encoding(vocab)
    datalist = enc.encode(raw_dataset)
    rands = torch.randint(0, len(datalist) - context_length, (batch_size,))
    for i in range(batch_size):
        tempx = []
        tempy = []
        start = int(rands[i].item())
        for j in range(context_length):
            tempx.append(datalist[start+j])
            tempy.append(datalist[start+j+1])
        X.append(tempx)
        Y.append(tempy)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X,Y

def make_tokens(raw_dataset: str):
    enc = tiktoken.get_encoding(vocab)
    return torch.tensor(enc.encode(raw_dataset), dtype=torch.long)

def batch_loader_stride_tokens(tokens: torch.Tensor, context_length: int, batch_size: int, stride: int, position: int):
    windows = tokens.unfold(0, context_length + 1, stride)
    Nw = windows.size(0)
    idx = (torch.arange(position, position + batch_size) % Nw)
    w = windows[idx]
    return w[:, :-1], w[:, 1:]
