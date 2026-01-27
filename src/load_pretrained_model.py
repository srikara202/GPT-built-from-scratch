import torch
from GPT import GPT
from pathlib import Path
import pickle

my = GPT(
    vocab_size=50257,
    context_length=1024,
    model_dim=768,
    num_blocks=6,
    num_heads=12
)

def load_distilgpt2_into_my_gpt(my_model, device="cpu"):
    """
    Loads HuggingFace distilgpt2 weights into YOUR architecture by converting:
    - HF Conv1D weights (in, out) -> nn.Linear weights (out, in)
    - HF fused QKV (c_attn) -> per-head Q, K, V linears
    """
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    hf.eval()

    sd = hf.state_dict()

    # --- basic shape checks ---
    D = my_model.vocab_embedding.weight.shape[1]
    H = len(my_model.nxTransformer[0].mha.attention_heads)
    assert D % H == 0, "model_dim must be divisible by num_heads"
    hd = D // H

    # --- embeddings ---
    with torch.no_grad():
        my_model.vocab_embedding.weight.copy_(sd["transformer.wte.weight"])
        # pos emb: allow my context_length <= 1024
        T_my = my_model.pos_embedding.weight.shape[0]
        my_model.pos_embedding.weight.copy_(sd["transformer.wpe.weight"][:T_my])

    # --- blocks ---
    for i in range(len(my_model.nxTransformer)):
        block = my_model.nxTransformer[i]

        # LayerNorms
        with torch.no_grad():
            block.norm1.weight.copy_(sd[f"transformer.h.{i}.ln_1.weight"])
            block.norm1.bias.copy_(sd[f"transformer.h.{i}.ln_1.bias"])
            block.norm2.weight.copy_(sd[f"transformer.h.{i}.ln_2.weight"])
            block.norm2.bias.copy_(sd[f"transformer.h.{i}.ln_2.bias"])

        # Attention: HF fused QKV
        W = sd[f"transformer.h.{i}.attn.c_attn.weight"]  # (D, 3D) in HF Conv1D format (in, out)
        b = sd[f"transformer.h.{i}.attn.c_attn.bias"]    # (3D,)

        Wq, Wk, Wv = W.split(D, dim=1)  # each (D, D)
        bq, bk, bv = b.split(D, dim=0)  # each (D,)

        # Copy into your per-head Q/K/V linears
        for h in range(H):
            s, e = h * hd, (h + 1) * hd
            head = block.mha.attention_heads[h]

            with torch.no_grad():
                # nn.Linear weight is (out, in), so transpose (D, hd) -> (hd, D)
                head.query.weight.copy_(Wq[:, s:e].T)
                head.query.bias.copy_(bq[s:e])

                head.key.weight.copy_(Wk[:, s:e].T)
                head.key.bias.copy_(bk[s:e])

                head.value.weight.copy_(Wv[:, s:e].T)
                head.value.bias.copy_(bv[s:e])

        # Attention output projection: HF c_proj is Conv1D (in, out) => transpose for Linear
        Wp = sd[f"transformer.h.{i}.attn.c_proj.weight"]  # (D, D) (in, out)
        bp = sd[f"transformer.h.{i}.attn.c_proj.bias"]    # (D,)
        with torch.no_grad():
            block.mha.out_proj.weight.copy_(Wp.T)
            block.mha.out_proj.bias.copy_(bp)

        # MLP: c_fc (D -> 4D), c_proj (4D -> D), both Conv1D => transpose for Linear
        Wfc = sd[f"transformer.h.{i}.mlp.c_fc.weight"]   # (D, 4D) (in, out)
        bfc = sd[f"transformer.h.{i}.mlp.c_fc.bias"]     # (4D,)
        Wpr = sd[f"transformer.h.{i}.mlp.c_proj.weight"] # (4D, D) (in, out) because nx=4D, nf=D
        bpr = sd[f"transformer.h.{i}.mlp.c_proj.bias"]   # (D,)

        with torch.no_grad():
            block.feed_forward.up_projection.weight.copy_(Wfc.T)   # (4D, D)
            block.feed_forward.up_projection.bias.copy_(bfc)

            block.feed_forward.down_projection.weight.copy_(Wpr.T) # (D, 4D)
            block.feed_forward.down_projection.bias.copy_(bpr)

    # --- final LN ---
    with torch.no_grad():
        my_model.norm.weight.copy_(sd["transformer.ln_f.weight"])
        my_model.norm.bias.copy_(sd["transformer.ln_f.bias"])

    # --- lm head ---
    # HF lm_head is (vocab, D). Linear weight is also (vocab, D). Bias doesn't exist in HF.
    with torch.no_grad():
        my_model.linear.weight.copy_(sd["lm_head.weight"])
        if my_model.linear.bias is not None:
            my_model.linear.bias.zero_()

    my_model.to(device)
    my_model.eval()
    return my_model, hf

from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model, hf = load_distilgpt2_into_my_gpt(my, device=device)

tok = AutoTokenizer.from_pretrained("distilgpt2")
x = tok("Hello, I am a language model", return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    logits_hf = hf(x).logits
    logits_my = model(x)

print("max abs diff:", (logits_hf - logits_my).abs().max().item())
print("hf argmax:", logits_hf[0, -1].argmax().item())
print("my argmax:", logits_my[0, -1].argmax().item())


file_to_delete = Path("pre_trained_model.pkl")
file_to_delete.unlink(missing_ok=True)

with open("pre_trained_model.pkl", 'wb') as file:
    pickle.dump(model, file)