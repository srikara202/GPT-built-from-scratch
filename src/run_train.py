import torch
import tiktoken
import pickle
from GPT import GPT
from Training_Loop import train_gpt
import yaml
import time
from pathlib import Path

def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        # Use safe_load to avoid arbitrary code execution
        config = yaml.safe_load(file)
    return config

# Load the configuration
config_data = load_hyperparameters('config.yaml')
vocab = config_data['vocab']

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_text = read_text("train.txt")

    # Match tokenizer and vocab size
    enc = tiktoken.get_encoding(vocab)
    vocab_size = enc.n_vocab  # should be 50257 for gpt2

    # Build model (tune these for your GPU)
    model = GPT(
        vocab_size=vocab_size,
        context_length=256,
        model_dim=512,
        num_blocks=12,
        num_heads=8,
    )

    # Train
    t1 = time.time()
    train_gpt(
        model=model,
        raw_text=raw_text,
        batch_size=32,
        epochs=10,
        lr=3e-4,
        device=device,
        stride=model.context_length,  # or 1 for more overlap
    )
    t2 = time.time()
    print("time taken=", t2-t1)

    file_to_delete = Path("model.pkl")
    file_to_delete.unlink(missing_ok=True)

    file_to_delete = Path("gpt_weights.pt")
    file_to_delete.unlink(missing_ok=True)

    torch.save(model.state_dict(), "gpt_weights.pt")
    with open("model.pkl", 'wb') as file:
        pickle.dump(model, file)
    print("Saved: gpt_weights.pt")

if __name__ == "__main__":
    main()
