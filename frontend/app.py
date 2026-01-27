# app.py
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent          # .../project/frontend
PROJECT_ROOT = APP_DIR.parent                      # .../project
SRC_DIR = PROJECT_ROOT / "src"                     # .../project/src

sys.path.insert(0, str(SRC_DIR))


import os
import pickle
import yaml
from typing import Optional, Tuple, List, Dict
from GPT import GPT
import streamlit as st
import torch
import torch.nn.functional as F
import tiktoken

st.set_page_config(page_title="GPT From Scratch", page_icon="🤖", layout="centered")


# ----------------------------
# Load config / tokenizer / model
# ----------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path}. Put it next to app.py.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_tokenizer(vocab_name: str):
    enc = tiktoken.get_encoding(vocab_name)
    eos_id = enc.eot_token
    return enc, eos_id


def load_model(model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Put it next to app.py.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # move to device if it's a torch nn.Module
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


# ----------------------------
# Prompt builder (chat -> single text prompt)
# ----------------------------
def build_prompt(messages: List[Dict[str, str]]) -> str:
    # Same idea as before; tweak if you trained on a different format.
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        else:
            lines.append(f"Assistant: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ----------------------------
# Generation (matches your Generator.py logic)
# ----------------------------
@torch.no_grad()
def generate_text(
    model,
    enc,
    eos_id: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    seed: Optional[int] = 0,
) -> str:
    if hasattr(model, "eval"):
        model.eval()

    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

    prompt_tokens_1d = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)  # (T,)
    prompt_len = prompt_tokens_1d.numel()

    # context: (B, T)
    context = prompt_tokens_1d.unsqueeze(0)

    # Use model.context_length like your code
    context_length = getattr(model, "context_length", 128)

    g = None
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(int(seed))

    for _ in range(int(max_new_tokens)):
        x = context[:, -context_length:]            # (B, Tctx)
        logits = model(x)[:, -1, :]                 # (B, V)

        # temperature
        logits = logits / max(float(temperature), 1e-8)

        # optional top-k filtering (safe clamp)
        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k)
            cutoff = v[:, [-1]]
            logits = torch.where(logits < cutoff, torch.tensor(-float("inf"), device=device), logits)

        probs = F.softmax(logits, dim=-1)           # (B, V)
        next_tok = torch.multinomial(probs, 1, generator=g)  # (B, 1)

        if next_tok.item() == eos_id:
            break

        context = torch.cat([context, next_tok], dim=1)  # (B, T+1)

    new_token_ids = context[0, prompt_len:].tolist()   # only generated tokens
    return enc.decode(new_token_ids).strip()


# ----------------------------
# UI
# ----------------------------
st.title("🤖 GPT-2 reproduced from scratch")
# st.caption("A mini GPT built and trained on a local laptop")

with st.sidebar:
    st.header("Generation Settings")
    temperature = st.slider("Temperature", 0.05, 2.0, 0.8, 0.05)
    top_k_ui = st.slider("Top-k (0 = off)", 0, 200, 50, 1)
    max_new_tokens = st.slider("Max new tokens", 1, 512, 200, 1)
    seed_ui = st.number_input("Seed (-1 = random)", value=0, step=1)

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []

# init state
if "messages" not in st.session_state:
    st.session_state.messages = []

# load assets
try:
    config = load_config("config.yaml")
    vocab_name = config["vocab"]
except Exception as e:
    st.error(f"Config error: {e}")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    enc, eos_id = load_tokenizer(vocab_name)
except Exception as e:
    st.error(f"Tokenizer error: {e}")
    st.stop()

try:
    model = load_model("pre_trained_model.pkl", device)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# input
user_text = st.chat_input("Type a message...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            prompt = build_prompt(st.session_state.messages)

            top_k = None if top_k_ui == 0 else int(top_k_ui)
            seed = None if int(seed_ui) == -1 else int(seed_ui)

            reply = generate_text(
                model=model,
                enc=enc,
                eos_id=eos_id,
                prompt=prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=top_k,
                seed=seed,
            )
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
