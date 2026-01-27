import torch
import torch.nn as nn
import torch.nn.functional as F
from Data_Preprocessing import make_tokens

def train_gpt(model, raw_text: str, batch_size: int, epochs=5, lr=3e-4, device="cuda", stride=None):
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device).train()

    # --- tokenize once ---
    tokens = make_tokens(raw_text)  # expect (N,) LongTensor (CPU)
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.to(device)  # <<< BIG WIN: move once

    T = model.context_length
    stride = T if stride is None else stride

    # build windows on GPU (view)
    windows = tokens.unfold(0, T + 1, stride)  # (Nw, T+1) on GPU
    Nw = windows.size(0)
    if Nw == 0:
        raise ValueError("Not enough tokens to make even one (context_length+1) window.")

    # optimizer (try fused when available)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=(device == "cuda"))
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda',enabled=use_amp)

    for epoch in range(epochs):
        # shuffle windows each epoch (also helps quality)
        perm = torch.randperm(Nw, device=device)

        total_loss, steps = 0.0, 0
        for start in range(0, Nw, batch_size):
            idx = perm[start:start + batch_size]
            w = windows[idx]          # (B, T+1) already on GPU

            x = w[:, :-1]             # (B, T)
            y = w[:, 1:]              # (B, T)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda',enabled=use_amp, dtype=torch.float16):
                logits = model(x)     # (B, T, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch+1}/{epochs} | loss: {total_loss / steps:.4f}")