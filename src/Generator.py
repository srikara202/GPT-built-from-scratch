import torch
import torch.nn.functional as F
import tiktoken
import pickle
import yaml

def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        # Use safe_load to avoid arbitrary code execution
        config = yaml.safe_load(file)
    return config

# Load the configuration
config_data = load_hyperparameters('config.yaml')
vocab = config_data['vocab']


def generate_text(model, prompt: str, max_new_tokens: int, temperature: float = 1.5, top_k: int | None = None, seed: int | None = 0):

    enc = tiktoken.get_encoding(vocab)
    eos_id = enc.eot_token

    @torch.no_grad()
    def generate(model, context: torch.Tensor, max_new_tokens: int, temperature: float = 1.5, top_k: int | None = None, seed: int | None = 0):
        model.eval()
        device = next(model.parameters()).device

        # context: (T,) or (B, T) of token ids
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context = context.to(device)

        g = None
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)

        for _ in range(max_new_tokens):
            x = context[:, -model.context_length:]          # (B, Tctx)
            logits = model(x)[:, -1, :]                     # (B, V)

            # temperature
            logits = logits / max(temperature, 1e-8)

            # optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits = torch.where(logits < v[:, [-1]], torch.tensor(-float("inf"), device=device), logits)

            probs = F.softmax(logits, dim=-1)               # (B, V)
            next_tok = torch.multinomial(probs, 1, generator=g)  # (B, 1)
            text_so_far = enc.decode(context[0].tolist())
            if "\nUser:" in text_so_far[prompt_len:]:
                break
            context = torch.cat([context, next_tok], dim=1)      # (B, T+1)

        return context  # (B, original_T + max_new_tokens)

    # --- decoding example (tiktoken gpt2) ---
    prompt_tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)
    out_tokens = generate(model, prompt_tokens, max_new_tokens, temperature, top_k, seed)
    prompt_len = prompt_tokens.numel() if prompt_tokens.dim() == 1 else prompt_tokens.size(1)
    new_token_ids = out_tokens[0, prompt_len:].tolist()   # only generated tokens (no prompt)
    answer = enc.decode(new_token_ids)
    return answer

with open("model.pkl", 'rb') as file:
        model = pickle.load(file)
prompt = "hello there"
max_new_tokens = 120,
temperature = 1.0
top_k = 50
seed = 0
print(generate_text(model, prompt, max_new_tokens, temperature, top_k, seed))