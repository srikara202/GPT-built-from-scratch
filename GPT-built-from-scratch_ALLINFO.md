# GPT-built-from-scratch All-Info Documentation

Generated from repository evidence only. This document describes the repository as observed in the working tree at `C:\D Drive\Python Projects\GPT-built-from-scratch` on 2026-05-16.

Analysis scope:

- Tracked files were inventoried with `git ls-files`.
- Notable untracked files were checked with `git ls-files --others --exclude-standard`.
- `.git` internals were intentionally excluded.
- `train.txt` and `image.png` were documented at an appropriate level because `train.txt` is a large corpus file tracked with Git LFS and `image.png` is a binary screenshot.
- No application or source code was modified to create this document.

## 1. Executive Summary

This repository is an educational, engineering-focused implementation of a GPT-2-style decoder-only transformer in PyTorch. It manually implements the core model stack, including token embeddings, positional embeddings, causal self-attention, multi-head attention, transformer blocks, next-token training, sampling-based text generation, a Hugging Face DistilGPT2 weight-transfer script, and a Streamlit UI for interactive prompt-based generation. The project is best understood as a "learn and demonstrate the internals" repository rather than a production LLM service.

The problem it solves is educational and portfolio-oriented: many projects call a pretrained model through a high-level API, while this repository exposes the mechanics underneath that API boundary. It shows how raw text becomes token windows, how a decoder-only transformer converts tokens into logits, how causal masking prevents future-token leakage, how training uses shifted next-token targets, how autoregressive generation samples one token at a time, and how pretrained Hugging Face weights can be translated into a custom module layout.

The target users are developers, students, interview candidates, ML learners, and reviewers who want to understand transformer internals from code instead of treating an LLM as a black box. The repository also gives a candidate something concrete to discuss in interviews: tensor shapes, weight layouts, causal masks, model checkpoints, training loops, and a simple demo app.

The core workflow has two main paths. The scratch-training path reads `train.txt`, tokenizes it with the GPT-2 tokenizer, creates next-token windows, trains a custom GPT model, and saves `gpt_weights.pt` plus `model.pkl`. The pretrained-demo path constructs a DistilGPT2-shaped custom model, downloads Hugging Face `distilgpt2`, copies weights into this custom architecture, validates logits against the Hugging Face model on a sample prompt, saves `pre_trained_model.pkl`, and serves it through `frontend/app.py`.

The main technical achievement is not just that the model architecture is written in PyTorch. The stronger signal is that `src/load_pretrained_model.py` understands the mismatch between Hugging Face's fused Conv1D-style GPT-2 attention weights and this repository's separate per-head `query`, `key`, and `value` linear layers, then slices and transposes those weights into the custom implementation.

Interview pitch: "I built a compact GPT-2-style decoder-only transformer from scratch in PyTorch, including causal self-attention, multi-head attention, a pre-layernorm transformer block, next-token training, autoregressive sampling, and a Streamlit demo. The most interesting part is that I also wrote a DistilGPT2 weight loader that maps Hugging Face's fused QKV and Conv1D-style weights into my custom per-head PyTorch modules, then validates the custom model against the reference logits."

## 2. Project Metadata

| Field | Evidence-based value |
|---|---|
| Inferred project name | `GPT-built-from-scratch` |
| Inference basis | Repository directory name, README title, Streamlit page title, and project content all point to a GPT-from-scratch educational transformer project. |
| Repository type | Python ML / educational deep learning project with a Streamlit frontend demo. |
| Main languages | Python, Markdown, YAML. |
| Main framework | PyTorch. |
| UI framework | Streamlit, imported by `frontend/app.py`; not listed in `requirements.txt`. |
| Tokenization | `tiktoken` with `vocab: "gpt2"` from `config.yaml`. |
| External model integration | Hugging Face Transformers `AutoModelForCausalLM` and `AutoTokenizer` for `distilgpt2` in `src/load_pretrained_model.py`. |
| Package/build tool | `pip` with `requirements.txt`; no `pyproject.toml`, `setup.py`, `Pipfile`, Poetry, uv, or Conda environment file is tracked. |
| Runtime assumptions | Python runtime with PyTorch, Transformers, TorchTyping, tiktoken, plus missing-but-imported `streamlit` and `PyYAML`; CUDA is optional and detected at runtime. |
| Database/storage | No database. Storage is file-based: text corpus, pickled models, PyTorch state dicts, and image documentation. |
| Authentication/authorization | None visible in tracked code. |
| APIs/services | Hugging Face model download for `distilgpt2`; Streamlit app deployment URL appears in README. |
| Important entrypoints | `src/run_train.py`, `src/Generator.py`, `src/load_pretrained_model.py`, `frontend/app.py`. |
| Important config files | `config.yaml`, `requirements.txt`, `.gitattributes`, `.gitignore`. |
| Test commands found | None. No test framework files or test scripts are tracked. |
| Run commands found | `python src/run_train.py`, `python src/Generator.py`, `python src/load_pretrained_model.py`, `streamlit run frontend/app.py`, `pip install -r requirements.txt`. |
| Build commands found | None. This is not packaged as a build artifact. |
| Deployment clues | README links to a hosted Streamlit demo at `https://gpt-built-from-scratch-gg43jqmryyynvfbn9fz7ve.streamlit.app/`; no Streamlit cloud config file, Dockerfile, or CI/CD config is tracked. |
| License | MIT License in `LICENSE`. |
| Git LFS | `.gitattributes` tracks `*.pkl` and `train.txt` through Git LFS. `git lfs ls-files` reports `train.txt`. |

## 3. Quick Start Guide

### Prerequisites

Based on repository evidence, a developer needs:

- Python 3.10 or newer is likely needed because `src/Generator.py` uses `int | None` union type syntax. The repo does not state an exact Python version.
- `pip`.
- PyTorch.
- Hugging Face Transformers.
- TorchTyping.
- tiktoken.
- PyYAML, because the code imports `yaml`.
- Streamlit, if running `frontend/app.py`.
- Optional CUDA-capable GPU for faster training and DistilGPT2 comparison. Code falls back to CPU in `src/run_train.py`, `src/Training_Loop.py`, `src/load_pretrained_model.py`, and `frontend/app.py`.
- Internet access for the first run of `src/load_pretrained_model.py`, because `AutoModelForCausalLM.from_pretrained("distilgpt2")` and `AutoTokenizer.from_pretrained("distilgpt2")` download model/tokenizer assets when not cached.

### Install steps

From the repository root:

```bash
pip install -r requirements.txt
```

The tracked `requirements.txt` contains:

```text
torch
transformers
torchtyping
tiktoken
```

The README and imports show that two additional packages may be needed:

```bash
pip install streamlit PyYAML
```

This second command is not listed in `requirements.txt`; it is inferred from `frontend/app.py`, `src/Data_Preprocessing.py`, `src/Generator.py`, and `src/run_train.py`, all of which import `yaml`, and from `frontend/app.py`, which imports `streamlit`.

### Environment variables

No required environment variables are visible in tracked files. No `.env` file is tracked. `.gitignore` ignores `.env`, `.envrc`, `.venv`, and several virtual environment directories.

Secret handling note: no API keys, passwords, or tokens were copied into this document. The tracked code does not expose secret values.

### Development commands

Run scratch training from the repository root:

```bash
python src/run_train.py
```

Run standalone generation after `model.pkl` has been produced:

```bash
python src/Generator.py
```

Load Hugging Face DistilGPT2 weights into the custom architecture and save `pre_trained_model.pkl`:

```bash
python src/load_pretrained_model.py
```

Launch the Streamlit demo after `pre_trained_model.pkl` exists:

```bash
streamlit run frontend/app.py
```

Important current-working-directory assumption: the commands above should be run from the repository root. Several files open `config.yaml`, `train.txt`, `model.pkl`, or `pre_trained_model.pkl` by relative path.

### Build command

No build command is tracked. This repository is executed directly as Python scripts and a Streamlit app.

### Test command

No test command is tracked. No `tests/` folder, `pytest.ini`, unit tests, or CI test configuration were found.

### Common troubleshooting notes

| Symptom | Likely cause from repo evidence | Where to inspect |
|---|---|---|
| `ModuleNotFoundError: No module named 'yaml'` | `PyYAML` is imported but not listed in `requirements.txt`. | `src/Data_Preprocessing.py`, `src/run_train.py`, `src/Generator.py`, `frontend/app.py` |
| `ModuleNotFoundError: No module named 'streamlit'` | `streamlit` is imported but not listed in `requirements.txt`. | `frontend/app.py`, README setup note |
| `FileNotFoundError: Missing config.yaml` in Streamlit | App checks `os.path.exists("config.yaml")`, so it expects root working directory. | `frontend/app.py:29-33` |
| `FileNotFoundError: Missing pre_trained_model.pkl` in Streamlit | `frontend/app.py` expects the pretrained transferred pickle to exist. | `frontend/app.py:42-53`, `src/load_pretrained_model.py:128-132` |
| `model.pkl` missing in standalone generator | `src/Generator.py` loads `model.pkl` at import/script execution. | `src/Generator.py:66-73`, `src/run_train.py:62-64` |
| `Generator.py` fails around `range(max_new_tokens)` | The script assigns `max_new_tokens = 120,`, a one-element tuple, not an int. | `src/Generator.py:69` |
| `Generator.py` fails around `prompt_len` | Inner `generate()` reads `prompt_len` before the outer function assigns it. | `src/Generator.py:51-62` |
| Training raises `ValueError` about not enough tokens | Corpus token count is too short for `context_length + 1` and chosen stride. | `src/Training_Loop.py:20-23` |
| Top-k error if top-k exceeds vocabulary in standalone generator | `src/Generator.py` does not clamp `top_k`, while the Streamlit version does. | `src/Generator.py:45-47`, `frontend/app.py:111-115` |
| Git status reports LFS clean-filter errors for `train.txt` | `train.txt` is tracked with Git LFS; this working copy produced `Access is denied` errors under `.git/lfs/tmp` while checking status. | `.gitattributes`, `git lfs ls-files` |

## 4. What The Project Does

### Main user-facing features

- Provides a Streamlit UI for entering messages into a chat-like interface.
- Maintains a session-local list of user and assistant messages in `st.session_state.messages`.
- Converts the message history into a text continuation prompt of the form `User: ...` and `Assistant: ...`.
- Exposes generation controls in a sidebar:
  - temperature from `0.05` to `2.0`;
  - top-k from `0` to `200`, where `0` disables top-k;
  - max new tokens from `1` to `512`;
  - seed where `-1` means random.
- Loads `pre_trained_model.pkl` and samples text from the custom model.
- Includes `image.png`, a screenshot of the Streamlit demo, referenced by the README.

The UI is intentionally not a production chatbot. README explicitly states the underlying model is still text continuation and the app is not chat-tuned or instruction-tuned.

### Main developer-facing features

- Manual GPT-like architecture in `src/GPT.py`, `src/Transformer_Block.py`, `src/Multi_Headed_Attention.py`, and `src/Self_Attention_Mechanism.py`.
- Training helpers in `src/Data_Preprocessing.py` and `src/Training_Loop.py`.
- Scripted scratch training in `src/run_train.py`.
- Scripted standalone generation in `src/Generator.py`.
- Hugging Face DistilGPT2 weight import and validation in `src/load_pretrained_model.py`.
- A README that already explains the educational purpose, architecture, run commands, and limitations.

### Inputs and outputs

| Flow | Inputs | Outputs |
|---|---|---|
| Scratch training | `train.txt`, `config.yaml`, model hyperparameters inside `src/run_train.py` | `gpt_weights.pt`, `model.pkl`, printed epoch loss |
| Standalone generation | `model.pkl`, hard-coded prompt/settings in `src/Generator.py`, `config.yaml` | Printed generated text |
| DistilGPT2 weight loading | Hugging Face `distilgpt2`, custom GPT architecture shape, sample validation prompt | Printed max logit difference and argmax tokens, `pre_trained_model.pkl` |
| Streamlit demo | `config.yaml`, `pre_trained_model.pkl`, user chat input, sidebar settings | Streamlit chat messages and generated reply |

### Important screens, pages, endpoints, commands, and jobs

- Screen: Streamlit single-page app in `frontend/app.py`.
- Commands:
  - `python src/run_train.py`
  - `python src/Generator.py`
  - `python src/load_pretrained_model.py`
  - `streamlit run frontend/app.py`
- No HTTP API endpoints are implemented by the repo beyond Streamlit's framework-managed app server.
- No background workers, schedulers, queues, or cron jobs are tracked.

### Expected happy paths

Scratch training happy path:

1. Install dependencies.
2. Run `python src/run_train.py` from repo root.
3. The script reads `train.txt`, detects CUDA if available, builds a model with context length 256, model dimension 512, 12 blocks, and 8 heads.
4. `train_gpt()` tokenizes once, unfolds tokens into windows, shuffles each epoch, trains with AdamW and cross-entropy, prints loss.
5. The script deletes prior `model.pkl` and `gpt_weights.pt` if present, saves the new state dict and pickle.

Pretrained demo happy path:

1. Install dependencies, including Transformers and network access or cached assets.
2. Run `python src/load_pretrained_model.py`.
3. It builds a DistilGPT2-shaped custom model, downloads/loads Hugging Face DistilGPT2, copies embeddings, norms, attention, MLP, and head weights, validates logits on a prompt, and saves `pre_trained_model.pkl`.
4. Run `streamlit run frontend/app.py` from repo root.
5. The UI loads the config, tokenizer, pickle, and generates replies from user input.

### Important failure paths

- Missing dependencies: `streamlit` and `PyYAML` may not be installed by `requirements.txt`.
- Missing model artifacts: `model.pkl` and `pre_trained_model.pkl` are generated outputs and are not tracked in the current file inventory.
- Network/caching failure: `load_pretrained_model.py` depends on Hugging Face `distilgpt2`.
- Short corpus: `Training_Loop.train_gpt()` raises a `ValueError` if no `context_length + 1` windows can be made.
- Path assumptions: scripts assume root-relative files and are not robust to arbitrary working directories.
- Standalone generator pitfalls: `prompt_len` is used before assignment inside the nested generator, and `max_new_tokens = 120,` creates a tuple.
- Pickle risk: both `src/Generator.py` and `frontend/app.py` load pickled model objects. Pickle is unsafe for untrusted files.

### Known limitations visible from the code

- No automated tests.
- No pinned dependency versions.
- No packaged module structure or relative imports.
- No CLI argument parsing; scripts use hard-coded model sizes, prompt text, paths, and generation defaults.
- No robust checkpoint metadata format.
- No authentication, rate limiting, or API serving layer.
- No training/validation split, evaluation loop, checkpoint resume, or metrics logging beyond printed loss.
- No Dockerfile, deployment manifest, or CI/CD config.
- Streamlit UI is a text-continuation wrapper, not a true chat/instruction model.

## 5. High-Level Architecture

### Major modules

| Module/file | Responsibility |
|---|---|
| `src/Self_Attention_Mechanism.py` | Implements one causal self-attention head with separate key/query/value projections. |
| `src/Multi_Headed_Attention.py` | Runs multiple `SingleHeadAttention` heads, concatenates their outputs, and applies output projection. |
| `src/Transformer_Block.py` | Implements a pre-layernorm transformer block with residual attention and feed-forward sublayers. |
| `src/GPT.py` | Assembles embeddings, repeated transformer blocks, final norm, and vocabulary projection. |
| `src/Data_Preprocessing.py` | Loads YAML tokenizer config and creates token tensors/window batches. |
| `src/Training_Loop.py` | Performs next-token training with AdamW, cross-entropy, AMP on CUDA, and gradient clipping. |
| `src/run_train.py` | End-to-end scratch-training entrypoint. |
| `src/Generator.py` | Standalone text generation entrypoint for `model.pkl`. |
| `src/load_pretrained_model.py` | Imports DistilGPT2 weights into the custom architecture and saves `pre_trained_model.pkl`. |
| `frontend/app.py` | Streamlit demo UI for chat-like text continuation. |

### Architecture diagram

```text
                    config.yaml
                        |
                        v
                 tiktoken.get_encoding
                        |
          +-------------+-------------+
          |                           |
          v                           v
  Scratch training              Streamlit / generation
  src/run_train.py              frontend/app.py or src/Generator.py
          |                           |
          v                           v
      train.txt                 prompt text / messages
          |                           |
          v                           v
  Data_Preprocessing.make_tokens  encoded prompt tokens
          |                           |
          v                           v
   windows x/y batches       autoregressive context crop
          |                           |
          +-------------+-------------+
                        |
                        v
                   GPT.forward
                        |
      token embeddings + positional embeddings
                        |
                        v
        N x TransformerBlock(pre-LN residual)
                        |
                        v
        final LayerNorm -> Linear vocab logits
                        |
          +-------------+-------------+
          |                           |
          v                           v
 cross_entropy training loss    softmax + multinomial sample
```

### Model internals

```text
context token ids (B, T)
  |
  +--> vocab_embedding(context)      -> (B, T, D)
  +--> pos_embedding(0..T-1)         -> (1, T, D)
  |
  v
embedded sequence                    -> (B, T, D)
  |
  v
TransformerBlock repeated num_blocks times:
  LayerNorm
  MultiHeadedSelfAttention:
    SingleHeadAttention x num_heads:
      Q = Linear(x), K = Linear(x), V = Linear(x)
      scores = Q K^T / sqrt(head_dim)
      causal lower-triangular mask
      weights = softmax(masked scores)
      head output = weights V
    concat heads
    output projection
  residual add
  LayerNorm
  feed-forward D -> 4D -> D with GELU/dropout
  residual add
  |
  v
final LayerNorm
  |
  v
Linear(D, vocab_size)
  |
  v
logits (B, T, V)
```

### Frontend/backend split

There is no separate backend service. The Streamlit app runs in one Python process and directly loads the pickled PyTorch model. It imports model classes from `src/` by inserting `SRC_DIR` into `sys.path`.

### Database/storage layer

No database is present. Storage is file-based:

- `train.txt`: local corpus for scratch training.
- `model.pkl`: generated pickle from scratch training, not currently tracked.
- `gpt_weights.pt`: generated PyTorch state dict from scratch training, not currently tracked.
- `pre_trained_model.pkl`: generated pickle from DistilGPT2 weight transfer, not currently tracked.
- `image.png`: tracked screenshot used by README.

### Authentication/authorization

No authentication or authorization code is present.

### AI/ML/LLM components

- Custom decoder-only transformer architecture.
- GPT-2 tokenizer through `tiktoken`.
- Hugging Face DistilGPT2 model/tokenizer loading for weight transfer and validation.
- Sampling controls: temperature, top-k, seed, max tokens.

### Background jobs

None visible. Training and weight loading are foreground scripts.

### External API/service calls

`src/load_pretrained_model.py` calls:

- `AutoModelForCausalLM.from_pretrained("distilgpt2")`
- `AutoTokenizer.from_pretrained("distilgpt2")`

These use Hugging Face Transformers model-loading behavior. The repo does not define API keys or private model names.

### Error handling/logging/observability

- `frontend/app.py` catches config, tokenizer, and model load exceptions and displays Streamlit errors.
- `src/Training_Loop.py` raises a `ValueError` when there are not enough tokens for one training window.
- Training logs only epoch-average loss via `print`.
- Weight loading logs only max absolute logit difference and argmax IDs.
- There is no structured logging, metrics backend, tracing, or monitoring configuration.

## 6. End-to-End Workflows

### Workflow 1: Scratch training from local corpus

| Aspect | Details |
|---|---|
| Trigger | Developer runs `python src/run_train.py` from repo root. |
| Files involved | `src/run_train.py`, `src/Training_Loop.py`, `src/Data_Preprocessing.py`, `src/GPT.py`, `src/Transformer_Block.py`, `src/Multi_Headed_Attention.py`, `src/Self_Attention_Mechanism.py`, `config.yaml`, `train.txt`. |
| Inputs | `train.txt`, `config.yaml`, hard-coded model hyperparameters. |
| Outputs | Printed epoch losses, printed elapsed time, `gpt_weights.pt`, `model.pkl`. |
| Side effects | Deletes existing `model.pkl` and `gpt_weights.pt` before saving replacements. |

Step-by-step flow:

1. `src/run_train.py` imports `GPT` and `train_gpt`.
2. `load_hyperparameters("config.yaml")` reads YAML and extracts `vocab`.
3. `main()` chooses `cuda` if available else `cpu`.
4. `read_text("train.txt")` reads the corpus using UTF-8 with replacement for invalid bytes.
5. `tiktoken.get_encoding(vocab)` creates the tokenizer; current `vocab` is `gpt2`.
6. `enc.n_vocab` sets `vocab_size`.
7. `GPT(...)` creates a model with `context_length=256`, `model_dim=512`, `num_blocks=12`, `num_heads=8`.
8. `train_gpt()` moves the model to device, tokenizes the full text once with `make_tokens()`, moves tokens to device, and unfolds token windows of length `T + 1`.
9. Each epoch shuffles window indices using `torch.randperm`.
10. Each batch splits windows into input `x = w[:, :-1]` and target `y = w[:, 1:]`.
11. The model returns logits `(B, T, V)`.
12. Cross-entropy flattens logits and targets to train next-token prediction.
13. AMP is enabled only on CUDA. Gradients are scaled, unscaled, clipped to norm `1.0`, and stepped through AdamW.
14. After training, the script removes old artifacts and saves a state dict plus pickle.

Failure behavior:

- If `train.txt` is missing, `read_text()` raises `FileNotFoundError`.
- If `config.yaml` is missing or has no `vocab`, config loading or dictionary indexing fails.
- If the corpus is too short, `train_gpt()` raises `ValueError`.
- If dependencies are missing, imports fail.
- If memory is insufficient, PyTorch may raise CUDA or CPU memory errors; no custom recovery exists.

Where to look:

- `src/run_train.py:24-68`
- `src/Training_Loop.py:6-64`
- `src/Data_Preprocessing.py:38-47`

### Workflow 2: Standalone generation from scratch-trained model

| Aspect | Details |
|---|---|
| Trigger | Developer runs `python src/Generator.py` from repo root. |
| Files involved | `src/Generator.py`, `config.yaml`, generated `model.pkl`, model modules needed by pickle. |
| Inputs | `model.pkl`, hard-coded prompt `"hello there"`, hard-coded sampling settings. |
| Outputs | Printed generated text if script executes successfully. |
| Side effects | None intended. It reads a pickle and prints output. |

Step-by-step flow:

1. `src/Generator.py` loads `config.yaml` and extracts `vocab`.
2. `generate_text()` creates a tiktoken encoder and reads `eot_token`.
3. The nested `generate()` function sets the model to eval mode and moves context to the model device.
4. If a seed is provided, it creates a `torch.Generator` on the model device.
5. For each requested token, it crops context to `model.context_length`, obtains final-position logits, divides by temperature, optionally applies top-k filtering, softmaxes, samples one token, checks text for `"\nUser:"`, and appends the token.
6. The outer function decodes generated tokens after the prompt.
7. Script-level code loads `model.pkl`, defines prompt/settings, and prints the generated answer.

Important caveat:

- As currently written, the nested `generate()` reads `prompt_len` before the outer function assigns it. Also `max_new_tokens = 120,` creates a tuple. These are likely runtime bugs in the standalone script. The Streamlit implementation has a cleaner generation function and computes `prompt_len` before the loop.

Where to look:

- `src/Generator.py:18-64`
- `src/Generator.py:66-73`

### Workflow 3: DistilGPT2 weight transfer and validation

| Aspect | Details |
|---|---|
| Trigger | Developer runs `python src/load_pretrained_model.py` from repo root. |
| Files involved | `src/load_pretrained_model.py`, `src/GPT.py`, architecture modules, generated `pre_trained_model.pkl`. |
| Inputs | Hugging Face `distilgpt2` model and tokenizer, hard-coded validation prompt. |
| Outputs | Printed max absolute difference and argmax token IDs, `pre_trained_model.pkl`. |
| Side effects | Downloads/caches Hugging Face assets if absent; deletes prior `pre_trained_model.pkl`; writes a new pickle. |

Step-by-step flow:

1. A custom `GPT` instance named `my` is created with DistilGPT2-compatible dimensions: vocab 50257, context length 1024, model dimension 768, 6 blocks, and 12 heads.
2. `load_distilgpt2_into_my_gpt()` imports `AutoModelForCausalLM`, loads `distilgpt2`, and gets its state dict.
3. The function reads the custom model dimension `D`, number of heads `H`, asserts divisibility, and computes head dimension `hd`.
4. Token and positional embeddings are copied directly, with positional embeddings sliced to the custom context length.
5. For each transformer block:
   - LayerNorm weights and biases are copied.
   - Hugging Face fused attention input weights `c_attn.weight` and bias are split into Q, K, and V.
   - Each Q/K/V matrix is sliced per head.
   - Each head slice is transposed into `nn.Linear` layout and copied into `head.query`, `head.key`, and `head.value`.
   - Attention output projection weights are transposed and copied.
   - MLP `c_fc` and `c_proj` weights are transposed and copied into the feed-forward layers.
6. Final layer norm weights and language model head weights are copied.
7. The custom model is moved to device and set to eval mode.
8. The script tokenizes `"Hello, I am a language model"`, runs both Hugging Face and custom models, prints max absolute logit difference, prints final-position argmax token IDs, deletes any existing `pre_trained_model.pkl`, and pickles the custom model.

Failure behavior:

- Missing `transformers` dependency fails import.
- No internet and no cache can fail `from_pretrained`.
- Any mismatch between custom architecture shape and DistilGPT2 state dict shape fails during `copy_`.
- Pickle writing can fail if filesystem permissions are insufficient.

Where to look:

- `src/load_pretrained_model.py:6-12`
- `src/load_pretrained_model.py:14-108`
- `src/load_pretrained_model.py:110-132`

### Workflow 4: Streamlit interactive demo

| Aspect | Details |
|---|---|
| Trigger | Developer runs `streamlit run frontend/app.py` from repo root. |
| Files involved | `frontend/app.py`, `config.yaml`, generated `pre_trained_model.pkl`, `src/GPT.py` and architecture modules. |
| Inputs | User chat input, sidebar generation settings, `pre_trained_model.pkl`, tokenizer config. |
| Outputs | Streamlit-rendered chat history and model-generated reply. |
| Side effects | Updates `st.session_state.messages` in the current Streamlit session. |

Step-by-step flow:

1. `frontend/app.py` computes `APP_DIR`, `PROJECT_ROOT`, and `SRC_DIR`, then inserts `src` into `sys.path`.
2. Streamlit page config is set.
3. Helper functions are defined:
   - `load_config()`
   - `load_tokenizer()`
   - `load_model()`
   - `build_prompt()`
   - `generate_text()`
4. The UI renders a title, captions, sidebar sliders, seed input, and clear-chat button.
5. Session state initializes `messages` to an empty list.
6. The app loads `config.yaml`, creates a tokenizer, and loads `pre_trained_model.pkl` onto CUDA or CPU.
7. Existing messages are rendered using `st.chat_message`.
8. On new user input, the app appends a user message, builds a prompt from all messages, maps top-k and seed UI values to generation parameters, calls `generate_text()`, writes the reply, and appends the assistant reply.

Failure behavior:

- Config/tokenizer/model loading exceptions are caught and shown with `st.error()`, then `st.stop()` is called.
- Generation exceptions inside user-input handling are not separately caught.
- If `pre_trained_model.pkl` is untrusted, pickle loading is unsafe.

Where to look:

- `frontend/app.py:5-22`
- `frontend/app.py:29-53`
- `frontend/app.py:59-68`
- `frontend/app.py:74-126`
- `frontend/app.py:132-203`

## 7. Data Model, State, And Configuration

### Model and tensor data structures

| Structure | Shape / type | Created in | Purpose |
|---|---|---|---|
| Raw corpus | Python `str` | `src/run_train.py:20-27` | Training text loaded from `train.txt`. |
| Token tensor | `torch.LongTensor` of shape `(N,)` | `src/Data_Preprocessing.py:38-40` | Full encoded corpus. |
| Random batch `X` | `torch.LongTensor` of shape `(B, T)` | `src/Data_Preprocessing.py:17-36` | Input context tokens for old/random loader. |
| Random batch `Y` | `torch.LongTensor` of shape `(B, T)` | `src/Data_Preprocessing.py:17-36` | Shifted target tokens for old/random loader. |
| Unfolded windows | Tensor shape `(Nw, T+1)` | `src/Training_Loop.py:20` | Adjacent input+target slices. |
| Training input `x` | Tensor shape `(B, T)` | `src/Training_Loop.py:43` | Context tokens. |
| Training target `y` | Tensor shape `(B, T)` | `src/Training_Loop.py:44` | Next-token targets. |
| Token embeddings | Tensor shape `(B, T, D)` | `src/GPT.py:23` | Learned representation of token IDs. |
| Position embeddings | Tensor shape `(1, T, D)` after lookup | `src/GPT.py:22-23` | Learned position information. |
| Logits | Tensor shape `(B, T, V)` | `src/GPT.py:26-28` | Raw vocabulary scores. |
| Attention scores | Tensor shape `(B, T, T)` per head | `src/Self_Attention_Mechanism.py:18-19` | Query-key similarity. |
| Causal mask | Tensor shape `(T, T)` boolean | `src/Self_Attention_Mechanism.py:23-25` | Blocks future positions. |
| Streamlit message | `dict` with `role` and `content` | `frontend/app.py:180`, `frontend/app.py:203` | Conversation history item. |

### Configuration

`config.yaml` contains one key:

```yaml
vocab: "gpt2"
```

Effects:

- `src/Data_Preprocessing.py` uses it at import time to set module-level `vocab`.
- `src/run_train.py` uses it to select tiktoken encoding and vocabulary size.
- `src/Generator.py` uses it to select tiktoken encoding.
- `frontend/app.py` uses it to select tiktoken encoding.

Risk:

- Several modules read `config.yaml` at import time or runtime using a root-relative path. Moving the working directory can break imports/scripts.

### State management

- PyTorch model state is held in module parameters.
- Training optimizer state is local inside `train_gpt()` and is not saved.
- Scratch model artifacts:
  - `gpt_weights.pt`: state dict only.
  - `model.pkl`: entire pickled model object.
- Pretrained model artifact:
  - `pre_trained_model.pkl`: entire pickled model object.
- Streamlit UI state:
  - `st.session_state.messages`: list of message dictionaries for the current web session.

### Validation rules

Explicit validation in code:

- `src/Training_Loop.py` checks `Nw == 0` and raises a descriptive `ValueError`.
- `src/load_pretrained_model.py` asserts `D % H == 0`.
- `frontend/app.py` checks existence of `config.yaml` and `pre_trained_model.pkl`.
- `frontend/app.py` clamps top-k to the vocabulary dimension.

Implicit requirements:

- `model_dim` must be divisible by `num_heads`; only the pretrained loader asserts this.
- `context.shape[-1]` must be less than or equal to `context_length`, or position embedding lookup will exceed `pos_embedding` size. Generation crops context; direct callers of `GPT.forward()` must handle this themselves.
- `top_k` should be positive and no greater than vocabulary size. Streamlit handles this; standalone generator does not.
- Temperature should not be zero. Both generation implementations divide by `max(temperature, 1e-8)`.

### File formats

| File | Format | Role |
|---|---|---|
| `config.yaml` | YAML | Tokenizer vocabulary configuration. |
| `requirements.txt` | Plain text pip requirements | Dependency list. |
| `train.txt` | Large text corpus, tracked with Git LFS | Local scratch-training corpus. |
| `image.png` | PNG image | Streamlit demo screenshot. |
| `*.pkl` generated | Python pickle, Git LFS configured | Whole model serialization. |
| `*.pt` generated | PyTorch state dict | Scratch-trained weights. |
| `.gitattributes` | Git attributes | LFS rules. |
| `.gitignore` | Git ignore patterns | Generated/cache/secrets exclusions. |

### Feature flags

No feature flag system is present. Behavior is controlled by:

- hard-coded script parameters,
- Streamlit sidebar settings,
- `config.yaml` tokenizer setting,
- CUDA availability.

## 8. API, Routes, Commands, And Entrypoints

### `src/run_train.py`

Type: script entrypoint.

Called by: developer command `python src/run_train.py`.

Calls:

- `load_hyperparameters("config.yaml")`
- `read_text("train.txt")`
- `tiktoken.get_encoding(vocab)`
- `GPT(...)`
- `train_gpt(...)`
- `torch.save(...)`
- `pickle.dump(...)`

Produces:

- Printed training duration and epoch losses.
- `gpt_weights.pt`.
- `model.pkl`.

### `src/Generator.py`

Type: script entrypoint and generation helper.

Called by: developer command `python src/Generator.py`; `generate_text()` could also be imported, but script-level pickle loading runs immediately because it is not guarded by `if __name__ == "__main__"`.

Calls:

- `load_hyperparameters("config.yaml")`
- `tiktoken.get_encoding(vocab)`
- loaded model's `forward()`
- `torch.multinomial`
- `pickle.load`

Produces:

- Printed generated text, assuming current runtime issues are fixed or avoided.

### `src/load_pretrained_model.py`

Type: script entrypoint plus reusable loader function.

Called by: developer command `python src/load_pretrained_model.py`.

Calls:

- `GPT(...)`
- `AutoModelForCausalLM.from_pretrained("distilgpt2")`
- `AutoTokenizer.from_pretrained("distilgpt2")`
- many `copy_()` operations into custom model parameters
- `pickle.dump(...)`

Produces:

- Printed validation numbers.
- `pre_trained_model.pkl`.

### `frontend/app.py`

Type: Streamlit app entrypoint.

Called by: `streamlit run frontend/app.py`.

Calls:

- `load_config("config.yaml")`
- `load_tokenizer(vocab_name)`
- `load_model("pre_trained_model.pkl", device)`
- `build_prompt(st.session_state.messages)`
- `generate_text(...)`
- Streamlit UI primitives.

Produces:

- Interactive web page with chat-like input/output and sidebar generation controls.

### Package exports

The repository is not packaged. There is no `__init__.py`. Public surface is de facto:

- `GPT.GPT`
- `Transformer_Block.TransformerBlock`
- `Transformer_Block.TransformerBlock.FeedForward`
- `Multi_Headed_Attention.MultiHeadedSelfAttention`
- `Self_Attention_Mechanism.SingleHeadAttention`
- `Data_Preprocessing.load_hyperparameters`
- `Data_Preprocessing.batch_loader`
- `Data_Preprocessing.make_tokens`
- `Data_Preprocessing.batch_loader_stride_tokens`
- `Training_Loop.train_gpt`
- `run_train.read_text`
- `run_train.main`
- `Generator.generate_text`
- `load_pretrained_model.load_distilgpt2_into_my_gpt`
- `frontend.app` helper functions when loaded by Streamlit.

## 9. Full Repository Map

```text
GPT-built-from-scratch/
|-- .gitattributes
|-- .gitignore
|-- LICENSE
|-- README.md
|-- config.yaml
|-- frontend/
|   `-- app.py
|-- image.png
|-- requirements.txt
|-- src/
|   |-- Data_Preprocessing.py
|   |-- GPT.py
|   |-- Generator.py
|   |-- Multi_Headed_Attention.py
|   |-- Self_Attention_Mechanism.py
|   |-- Training_Loop.py
|   |-- Transformer_Block.py
|   |-- load_pretrained_model.py
|   `-- run_train.py
`-- train.txt
```

### Folder purposes

| Folder | Purpose |
|---|---|
| Repository root | Project documentation, config, dependency list, corpus, screenshot, Git metadata rules, and top-level generated output paths expected by scripts. |
| `frontend/` | Streamlit UI for interactive prompt-based generation using `pre_trained_model.pkl`. |
| `src/` | Core model, training, generation, preprocessing, and pretrained weight-loading scripts. |

### Tracked file one-line summaries

| File | One-line purpose |
|---|---|
| `.gitattributes` | Configures Git LFS for `*.pkl` artifacts and `train.txt`. |
| `.gitignore` | Ignores Python caches, build outputs, virtual environments, local secrets, generated artifacts, and editor/tool caches. |
| `LICENSE` | MIT license grant and disclaimer. |
| `README.md` | Primary project overview, architecture explanation, setup, run commands, limitations, and demo screenshot reference. |
| `config.yaml` | Selects the GPT-2 tokenizer vocabulary for tiktoken. |
| `frontend/app.py` | Streamlit chat-style demo around the custom GPT model. |
| `image.png` | Screenshot of the Streamlit demo UI. |
| `requirements.txt` | Minimal Python dependency list for core ML code. |
| `src/Data_Preprocessing.py` | YAML config loading, tokenization, random batches, and stride-window batch helpers. |
| `src/GPT.py` | Top-level decoder-only transformer model. |
| `src/Generator.py` | Standalone autoregressive generation script for `model.pkl`. |
| `src/Multi_Headed_Attention.py` | Multi-head attention wrapper around single-head causal attention modules. |
| `src/Self_Attention_Mechanism.py` | Single causal self-attention head. |
| `src/Training_Loop.py` | Scratch-training loop with AdamW, cross-entropy, AMP, and gradient clipping. |
| `src/Transformer_Block.py` | Pre-layernorm transformer block and feed-forward submodule. |
| `src/load_pretrained_model.py` | DistilGPT2-to-custom weight transfer, validation, and pickle saving. |
| `src/run_train.py` | Scratch training script that builds, trains, and saves a model. |
| `train.txt` | Large local demonstration corpus for scratch training, tracked with Git LFS. |

## 10. File-By-File Deep Dive

### `.gitattributes`

**Role:** Configures Git LFS tracking for large generated/model-related files.

**Why it matters:** The project works with large text and model artifacts. This file prevents large `*.pkl` files and `train.txt` from being stored as ordinary Git blobs when committed through normal LFS flow.

**Key dependencies/imports:** Git LFS.

**Exports/public surface:** None.

**Used by:** Git and Git LFS.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1 | `*.pkl filter=lfs diff=lfs merge=lfs -text` | Routes pickle files through Git LFS. | Files matching `*.pkl`. | LFS pointer/object behavior. | Changes Git storage behavior for matching files. | Generated `model.pkl` and `pre_trained_model.pkl` are not currently tracked but would match. |
| 2 | `train.txt filter=lfs diff=lfs merge=lfs -text` | Routes `train.txt` through Git LFS. | `train.txt`. | LFS pointer/object behavior. | Changes Git storage behavior for the corpus file. | `git lfs ls-files` reports `train.txt`; `git status --short` produced LFS clean-filter warnings in this working copy. |

**Potential interview talking points:**

- The repo anticipates large model/corpus artifacts and uses LFS rather than plain Git blobs.
- Generated pickle artifacts are configured for LFS if they are ever tracked.

**Possible improvements or risks:**

- Pickled model files can be large and unsafe to load from untrusted sources.
- `gpt_weights.pt` is not configured for LFS unless manually added; `.gitignore` currently comments out `# gpt_weights.pt`, so it is not actively ignored.

### `.gitignore`

**Role:** Defines untracked files and folders Git should ignore.

**Why it matters:** It protects local environments, caches, build output, test output, and secrets such as `.env` from accidental commits.

**Key dependencies/imports:** Git.

**Exports/public surface:** None.

**Used by:** Git status/add behavior.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-3 | Commented `train.txt`, `gpt_weights.pt`, `model.pkl` | Shows these were considered for ignore but are not currently ignored because lines are comments. | Generated/corpus file names. | No active ignore behavior. | None. | `train.txt` is tracked; generated `model.pkl` may appear untracked unless otherwise excluded by global rules. |
| 5 | `INTERVIEW_PREP_GPT2_FROM_SCRATCH.md` | Ignores a specific interview-prep markdown file. | That exact filename. | Git ignores it. | Prevents tracking. | This new ALLINFO file has a different name and is not ignored. |
| 7-58 | Python bytecode, extension, packaging, installer, coverage/test output | Ignores common Python generated outputs and test artifacts. | Caches, wheels, coverage folders. | Cleaner Git status. | None. | No test outputs were observed. |
| 64-78 | Django, Flask, Scrapy, Sphinx patterns | Ignores common framework artifacts. | Framework-specific generated/local files. | Cleaner Git status. | None. | No evidence the project uses these frameworks. |
| 80-141 | Build tools, notebooks, pyenv, pipenv, uv, poetry, pdm, pixi, PEP 582, Celery, SageMath | Ignores standard ecosystem artifacts and local environments. | Tool-generated files/folders. | Cleaner Git status. | None. | Broad template-style ignore. |
| 143-151 | Environment files and virtualenv folders | Ignores `.env`, `.envrc`, `.venv`, `env/`, `venv/`, etc. | Local secrets and env folders. | Prevents accidental secret/environment commits. | None. | Important for secret hygiene. |
| 153-213 | IDE, docs, type checker, Cython, Abstra, VS Code, Ruff, PyPI, Cursor, Marimo | Ignores local tool metadata and caches. | Tool-specific artifacts. | Cleaner Git status. | None. | Indicates normal Python/editor hygiene rather than app behavior. |

**Potential interview talking points:**

- The project uses a broad Python-oriented ignore template, including `.env`.
- Generated artifacts are partially considered but not consistently active in ignore rules.

**Possible improvements or risks:**

- Uncommenting generated model artifacts such as `model.pkl`, `pre_trained_model.pkl`, and `gpt_weights.pt` may prevent accidental commits if these are not meant to be versioned.
- If large `.pt` files are intended to be tracked, `.gitattributes` should include them.

### `LICENSE`

**Role:** Defines the repository license.

**Why it matters:** The MIT license permits reuse, modification, distribution, sublicensing, and selling copies subject to preserving the copyright and license notice.

**Key dependencies/imports:** None.

**Exports/public surface:** Legal permission terms.

**Used by:** Users, contributors, package consumers, employers/reviewers.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1 | MIT License heading | Identifies license family. | None. | License identity. | None. | Standard permissive license. |
| 3 | Copyright notice | Attributes copyright to `srikara202` in 2026. | None. | Ownership notice. | None. | Repo evidence only; no additional author metadata inspected. |
| 5-13 | Permission grant | Grants broad rights to use, copy, modify, merge, publish, distribute, sublicense, and sell. | Software copy. | Legal permission. | Requires notice preservation. | Standard MIT wording. |
| 15-21 | Warranty/liability disclaimer | Provides software "as is" without warranty and limits liability. | Software usage. | Risk allocation. | None. | Important for demos and educational code. |

**Potential interview talking points:**

- The code is explicitly open under a permissive license.

**Possible improvements or risks:**

- None specific from the license file.

### `README.md`

**Role:** Primary human-facing project documentation.

**Why it matters:** It is the strongest evidence source for the project's purpose, architecture, intended workflows, limitations, and demo deployment.

**Key dependencies/imports:** References PyTorch, Hugging Face Transformers, tiktoken, TorchTyping, Streamlit, YAML, and local scripts.

**Exports/public surface:** Documentation sections; no code exports.

**Used by:** Developers, reviewers, interviewers, and this ALLINFO analysis.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-7 | Title, overview, demo link | Defines project as a GPT-2-style decoder-only transformer manually implemented in PyTorch and links hosted Streamlit demo. | Project files and demo URL. | Reader orientation. | None. | Explicitly says it is not a production LLM platform. |
| 9-20 | Why the project exists | Frames the repo as a reconstruction project to understand internals. | Educational motivation. | Value proposition. | None. | Useful interview framing. |
| 22-33 | Key features | Lists manual architecture, training, generation, weight import, validation, and UI. | Repo features. | Feature inventory. | None. | Broad but consistent with code. |
| 35-87 | Architecture overview | Explains embeddings, transformer blocks, self-attention, multi-head attention, and feed-forward network. | Source modules. | Conceptual architecture. | None. | Matches code structure. |
| 89-132 | Model mechanics | Describes tokenization, forward pass, logits, causal mask, and next-token prediction. | `config.yaml`, source code. | Data/tensor flow explanation. | None. | Accurately notes no softmax inside model. |
| 134-188 | DistilGPT2 weight loading | Explains `load_pretrained_model.py`, shape choices, copied weights, attention mapping, Conv1D transposes, and validation. | Hugging Face and custom model code. | Weight-transfer explanation. | None. | Strongest technical section. |
| 190-220 | Training pipeline | Documents `Data_Preprocessing.py`, `Training_Loop.py`, and `run_train.py`; lists model size and artifacts. | Training scripts. | Run/training explanation. | None. | Notes educational/demo scale. |
| 222-238 | Text generation pipeline | Explains token-by-token generation and dependency on `model.pkl`. | `src/Generator.py`. | Generation explanation. | None. | README does not mention the current script bugs. |
| 240-264 | Streamlit demo | Explains the UI as a chat-style wrapper around text continuation and references `image.png`. | `frontend/app.py`, screenshot. | UI framing. | None. | Correctly avoids overclaiming chat tuning. |
| 266-287 | Project structure | Lists folders/files and one-line roles. | Repo tree. | Navigation aid. | None. | Does not include license/git config details. |
| 289-309 | Setup/install | Shows `pip install -r requirements.txt` and notes missing `streamlit` and `PyYAML`. | Requirements/imports. | Setup guidance. | None. | Important because requirements are incomplete for UI/YAML. |
| 311-376 | How to run | Provides scratch training, generation, pretrained loading, and Streamlit commands. | Script entrypoints. | Operational guide. | None. | No tests or deployment build commands. |
| 378-390 | Example workflows | Gives scratch-training and pretrained-demo sequences. | Run commands. | User workflow guidance. | None. | Good high-level demo path. |
| 392-408 | Limitations | Defines what the project is not and warns about output quality. | Project scope. | Expectation setting. | None. | Useful for interviews and demos. |
| 410-416 | Possible improvements | Lists dependency pinning, validation tests, checkpoint cleanup, config centralization, and visualization ideas. | Observed gaps. | Roadmap ideas. | None. | Aligns with code risks. |
| 418-431 | Tech stack and closing | Summarizes stack and core portfolio signal. | Repo evidence. | Final framing. | None. | Good concise project summary. |

**Potential interview talking points:**

- The README itself frames the project honestly: educational, not production.
- It highlights the strongest technical signal: custom weight transfer from Hugging Face DistilGPT2.
- It clarifies that the Streamlit UI is not chat-tuned.

**Possible improvements or risks:**

- Add exact Python version and dependency versions.
- Add test instructions once tests exist.
- Add known current runtime caveats for `src/Generator.py`.
- Add artifact-generation order and current-working-directory assumptions more explicitly.

### `config.yaml`

**Role:** Single-key YAML configuration selecting the tokenizer vocabulary.

**Why it matters:** It controls tiktoken encoding across preprocessing, training, generation, and Streamlit.

**Key dependencies/imports:** Loaded by PyYAML via `yaml.safe_load`.

**Exports/public surface:** `vocab: "gpt2"`.

**Used by:** `src/Data_Preprocessing.py`, `src/run_train.py`, `src/Generator.py`, `frontend/app.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1 | `vocab: "gpt2"` | Selects GPT-2 tokenizer vocabulary for `tiktoken.get_encoding`. | None. | Config dictionary key `vocab`. | None. | Changing this affects token IDs, vocab size, model output dimension expectations, and compatibility with DistilGPT2. |

**Potential interview talking points:**

- Tokenization is centralized in a tiny config file.
- GPT-2 tokenization is consistent with DistilGPT2 compatibility.

**Possible improvements or risks:**

- Expand config to include model dimensions, paths, training settings, and generation defaults.
- Validate config keys and values before using them.

### `frontend/app.py`

**Role:** Streamlit app that wraps the custom GPT model in a chat-like interface.

**Why it matters:** It is the main user-facing demo and shows the model can be interacted with through a web UI rather than only through scripts.

**Key dependencies/imports:**

- `sys`, `pathlib.Path`: insert `src/` into import path.
- `os`: file existence checks.
- `pickle`: load the model artifact.
- `yaml`: read `config.yaml`.
- `typing.Optional`, `Tuple`, `List`, `Dict`: annotations; `Tuple` is imported but not used.
- `GPT`: imported from `src`, but not directly used in code except for pickle class availability/import side effects.
- `streamlit`: UI framework.
- `torch`, `torch.nn.functional`: model execution, sampling.
- `tiktoken`: tokenizer.

**Exports/public surface:** Streamlit app module plus helper functions `load_config`, `load_tokenizer`, `load_model`, `build_prompt`, and `generate_text`.

**Used by:** Streamlit runtime via `streamlit run frontend/app.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-9 | Path bootstrap | Computes app, project, and `src` directories, then prepends `src` to `sys.path`. | `__file__`. | Python can import `GPT` and related modules. | Mutates `sys.path`. | Enables non-package source layout to work under Streamlit. |
| 12-20 | Imports | Loads stdlib, model, UI, tensor, and tokenizer dependencies. | Installed packages. | Imported symbols. | Import failure if dependencies missing. | `streamlit` and `yaml` are not in `requirements.txt`; `Tuple` and `GPT` appear unused after import. |
| 22 | `st.set_page_config` | Sets Streamlit page title, icon, and centered layout. | Static settings. | Streamlit page metadata. | Must be called early in Streamlit app. | Source uses an emoji icon; PowerShell displayed mojibake, but the UI screenshot shows intended visual icon. |
| 29-33 | `load_config` | Checks for and reads `config.yaml` with `yaml.safe_load`. | `config_path`, default `"config.yaml"`. | Config dictionary. | Reads filesystem. | Error message says "next to app.py", but code checks current working directory. |
| 36-39 | `load_tokenizer` | Creates tiktoken encoder and gets end-of-text token ID. | `vocab_name`. | `(enc, eos_id)`. | May fail if encoding name invalid. | Current config is `gpt2`. |
| 42-53 | `load_model` | Checks model file, unpickles it, moves to device if possible, sets eval mode if possible. | `model_path`, `device`. | Loaded model. | Reads pickle; moves model. | Pickle is unsafe for untrusted files. Error message says "next to app.py" but code uses cwd. |
| 59-68 | `build_prompt` | Converts message dicts into a plain text transcript ending with `Assistant:`. | List of `{"role", "content"}` dicts. | Prompt string. | None. | Any role other than `"user"` is treated as assistant. No escaping/sanitization. |
| 74-84 | `generate_text` signature and no-grad | Defines sampling function under `@torch.no_grad`. | Model, encoder, prompt, sampling params. | Generated string. | Disables gradient tracking. | Suitable for inference. |
| 85-97 | Generation setup | Sets eval mode, finds device, encodes prompt, stores prompt length, creates batch context, reads context length. | Prompt string and model. | Context tensor. | None beyond eval state. | Defaults context length to 128 if model lacks `context_length`. |
| 99-101 | Optional seeded generator | Creates deterministic `torch.Generator` when seed is not `None`. | `seed`, device. | Generator or `None`. | None. | UI maps `-1` to `None` for random behavior. |
| 103-123 | Autoregressive loop | Crops context, computes logits, applies temperature, applies clamped top-k, softmaxes, samples token, breaks on EOS, appends token. | Current context, sampling settings. | Updated context. | Repeated model calls. | Top-k is safely clamped; no repetition penalty or stop string. |
| 125-126 | Decode generated suffix | Slices off prompt tokens and decodes only new tokens. | Final context and prompt length. | Stripped string reply. | None. | Empty string possible if EOS sampled immediately. |
| 132-144 | Static UI and sidebar controls | Renders title/captions, sliders, seed input, and clear-chat button. | User interactions. | UI state values. | Clear button resets `st.session_state.messages`. | Caption has typo `"Foundationsl"` in source. |
| 146-148 | Session state init | Ensures message list exists. | Streamlit session. | `messages` list. | Mutates session state. | Per-session only. |
| 151-170 | Load config, tokenizer, model | Uses try/except blocks and `st.stop()` on failure. | Files and dependencies. | `config`, `vocab_name`, `device`, `enc`, `eos_id`, `model`. | Reads files, loads pickle, may move model to GPU. | Good user-facing failure display for startup dependencies. |
| 172-175 | Render history | Iterates messages and writes each in a chat container. | Session messages. | Visible chat history. | UI rendering. | Assumes message roles are valid for Streamlit chat. |
| 178-203 | Handle new user input | Appends user message, renders it, builds prompt, maps controls to generation params, generates reply, writes and stores reply. | User text and sidebar values. | Assistant message. | Mutates session state and performs model inference. | Generation errors are not caught here. |

**Potential interview talking points:**

- The UI deliberately uses prompt formatting instead of claiming the model is instruction-tuned.
- The generation loop uses standard temperature and top-k sampling.
- Top-k is clamped against vocabulary size in the UI implementation.
- Streamlit app uses session state to maintain a conversation transcript.

**Possible improvements or risks:**

- Add `streamlit` and `PyYAML` to `requirements.txt`.
- Avoid pickle for untrusted model loading, or document trust assumptions.
- Resolve paths relative to `PROJECT_ROOT` instead of current working directory.
- Remove unused imports.
- Add try/except around generation calls.
- Add prompt-length/context-length warnings for very long conversations.
- Fix UI text typo.

### `image.png`

**Role:** Binary PNG screenshot of the Streamlit demo.

**Why it matters:** It visually documents the UI and is embedded in the README.

**Key dependencies/imports:** None at runtime. Referenced by Markdown image syntax in `README.md`.

**Exports/public surface:** Visual artifact.

**Used by:** `README.md`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| Whole file | PNG image, 143,800 bytes, observed dimensions 1919x870 | Shows the Streamlit app with sidebar generation controls, chat input, and generated response. | Screenshot capture. | Rendered image in README/viewers. | None. | Binary file; detailed code-level analysis is not applicable. |

**Potential interview talking points:**

- The project includes a visual demo artifact, making it easier to explain the UI without running it.
- Screenshot shows generation controls and a chat-like interface.

**Possible improvements or risks:**

- Keep screenshot updated when UI changes.
- Consider adding alt text or a smaller optimized image if README load size matters.

### `requirements.txt`

**Role:** Minimal dependency list for installing core Python packages.

**Why it matters:** It is the only tracked dependency manifest.

**Key dependencies/imports:** `pip` requirements format.

**Exports/public surface:** Four package names.

**Used by:** Developers running `pip install -r requirements.txt`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1 | `torch` | Installs PyTorch. | pip resolver. | Tensor/model/training runtime. | Downloads package. | No version pin; CUDA/CPU variant depends on environment/index. |
| 2 | `transformers` | Installs Hugging Face Transformers. | pip resolver. | DistilGPT2 loading. | Downloads package/dependencies. | Needed by `src/load_pretrained_model.py`. |
| 3 | `torchtyping` | Installs tensor type annotation helper. | pip resolver. | `TensorType` annotations. | Downloads package. | Used only for annotations; not enforced at runtime here. |
| 4 | `tiktoken` | Installs tokenizer library. | pip resolver. | GPT-2 encoding. | Downloads package. | Used across preprocessing/generation/UI. |

**Potential interview talking points:**

- Dependency surface is intentionally small for the core educational model.
- The repo relies on PyTorch plus tokenizer/model-loading libraries rather than a high-level training framework.

**Possible improvements or risks:**

- Add `streamlit` and `PyYAML`.
- Pin versions for reproducibility.
- Document Python version.
- Consider separate extras or dev requirements if tests are added.

### `src/Data_Preprocessing.py`

**Role:** Loads tokenizer config and provides tokenization/batch construction helpers.

**Why it matters:** It converts raw text into next-token training inputs and targets. This is where the corpus becomes model-ready token IDs.

**Key dependencies/imports:**

- `torch`: tensors and random indices.
- `tiktoken`: GPT-2 tokenization.
- `typing.Tuple`, `typing.List`: annotations.
- `yaml`: config parsing.

**Exports/public surface:**

- `load_hyperparameters(yaml_file)`
- `batch_loader(raw_dataset, context_length, batch_size)`
- `make_tokens(raw_dataset)`
- `batch_loader_stride_tokens(tokens, context_length, batch_size, stride, position)`
- module-level `config_data` and `vocab`

**Used by:**

- `src/Training_Loop.py` imports `make_tokens`.
- The old/random `batch_loader` and `batch_loader_stride_tokens` are not referenced by tracked code but are available helpers.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-4 | Imports | Loads PyTorch, tiktoken, typing annotations, and YAML parser. | Installed packages. | Imported modules. | Import failure if missing. | `PyYAML` is not in `requirements.txt`. |
| 6-10 | `load_hyperparameters` | Opens a YAML file and parses it with `yaml.safe_load`. | Path string. | Config dictionary. | Reads filesystem. | Uses safe loader, which avoids arbitrary YAML object construction. |
| 12-14 | Module-level config load | Loads `config.yaml` and stores `vocab`. | Root-relative `config.yaml`. | Module-level tokenizer name. | Reads file at import time. | Import fails if cwd lacks `config.yaml`; tightens cwd assumptions. |
| 17-36 | `batch_loader` | Encodes raw text, selects random start offsets, builds `X` contexts and shifted `Y` targets with a fixed manual seed. | Raw text, context length, batch size. | `(X, Y)` long tensors. | Calls `torch.manual_seed(0)`, affecting global RNG. | Fails if `len(datalist) - context_length <= 0`; annotation says `Tuple[List[List[int]]]` but returns tensors. |
| 38-40 | `make_tokens` | Encodes raw text into a 1-D `torch.long` tensor. | Raw text. | Token tensor. | None. | Used by current training loop. |
| 42-47 | `batch_loader_stride_tokens` | Uses `tokens.unfold` to create sliding windows, wraps batch positions modulo number of windows, returns input and target slices. | Token tensor, context length, batch size, stride, position. | `(x, y)` tensors. | None. | If no windows exist, modulo by zero can fail; not used by current training loop. |

**Potential interview talking points:**

- The shifted target setup is standard next-token prediction: `x` is tokens `0..T-1`, `y` is `1..T`.
- `unfold` gives an efficient tensor view for sliding windows.
- Config-driven tokenizer choice keeps tokenization consistent across training and inference.

**Possible improvements or risks:**

- Avoid import-time file reads; pass config explicitly.
- Fix return type annotations.
- Avoid resetting global RNG in `batch_loader`.
- Add validation for short datasets.
- Add tests for token shifting and stride behavior.

### `src/GPT.py`

**Role:** Defines the top-level decoder-only transformer model.

**Why it matters:** This is the central neural network object used by training, pretrained weight loading, generation, and the UI.

**Key dependencies/imports:**

- `torch`: position tensor creation and tensor operations.
- `torch.nn`: module, embeddings, sequential stack, layer norm, linear projection.
- `torchtyping.TensorType`: type annotations.
- `TransformerBlock`: repeated decoder block.

**Exports/public surface:**

- `GPT(nn.Module)`
- `GPT.__init__`
- `GPT.forward`

**Used by:**

- `src/run_train.py`
- `src/load_pretrained_model.py`
- `frontend/app.py` indirectly and for pickle class loading
- Pickled model artifacts

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-4 | Imports | Loads PyTorch, annotations, and transformer block class. | Installed packages and local module path. | Imported symbols. | Import failure if `src` not on `sys.path`. | Files use direct imports, not package-relative imports. |
| 6 | `class GPT(nn.Module)` | Declares top-level model. | PyTorch module system. | Reusable model class. | None. | Name collides conceptually with GPT family, but local class is clear. |
| 8-18 | `__init__` | Stores context length; creates token and position embeddings; appends `num_blocks` transformer blocks into `nn.Sequential`; creates final layer norm and vocab projection. | `vocab_size`, `context_length`, `model_dim`, `num_blocks`, `num_heads`. | Initialized model parameters. | Allocates model tensors. | No assertion that `model_dim % num_heads == 0`; lower modules rely on integer division. Softmax is commented out, correctly leaving logits raw. |
| 21-28 | `forward` | Builds position IDs, sums token and position embeddings, runs transformer stack, normalizes, projects to vocab logits, returns logits. | `context` token IDs shaped `(B, T)`. | Logits shaped `(B, T, vocab_size)`. | None beyond computation. | If `T > context_length`, position embedding lookup fails; callers must crop or constrain context. |

**Potential interview talking points:**

- The model produces logits rather than probabilities, which is correct for cross-entropy and flexible for sampling.
- Position embeddings are learned and indexed by sequence positions.
- Blocks are stacked dynamically through `num_blocks`.
- This is a decoder-only design because each attention head applies a causal mask.

**Possible improvements or risks:**

- Add shape assertions or docstrings for `(B, T)` input and maximum context length.
- Assert `model_dim % num_heads == 0`.
- Use `nn.ModuleList` or `nn.Sequential(*blocks)` construction style if targeting older PyTorch versions where `.append` behavior might vary.
- Consider tied token embedding and output projection weights if aiming closer to GPT-2 architecture; current code copies `lm_head.weight` separately from HF.

### `src/Generator.py`

**Role:** Standalone text generation script for a scratch-trained `model.pkl`.

**Why it matters:** It demonstrates autoregressive sampling outside the Streamlit UI.

**Key dependencies/imports:**

- `torch`: tensors, no-grad, generator.
- `torch.nn.functional`: softmax.
- `tiktoken`: encoding/decoding.
- `pickle`: model loading.
- `yaml`: config parsing.

**Exports/public surface:**

- `load_hyperparameters`
- `generate_text`
- Script-level behavior that loads `model.pkl` and prints generation.

**Used by:** Developer command `python src/Generator.py`. No tracked file imports it.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-5 | Imports | Loads tensor, sampling, tokenizer, pickle, and YAML dependencies. | Installed packages. | Imported symbols. | Import failure if missing. | `yaml` requires PyYAML. |
| 7-11 | `load_hyperparameters` | Reads YAML config with `safe_load`. | YAML path. | Config dictionary. | Reads filesystem. | Same helper duplicated in multiple files. |
| 13-15 | Module-level config load | Reads `config.yaml` and stores `vocab`. | Root-relative config. | Tokenizer name. | Reads file at import time. | Importing this module from another cwd can fail. |
| 18-21 | `generate_text` setup | Creates tokenizer and gets EOS token ID. | Model, prompt, sampling settings. | Encoder and EOS ID. | None. | `eos_id` is assigned but not used in the standalone loop. |
| 23-35 | Nested `generate` setup | Defines no-grad generation helper; sets eval; moves context to model device; creates seeded generator. | Model, context, max tokens, temperature, top-k, seed. | Prepared context and RNG. | Model eval mode. | Depends on `next(model.parameters())`; fails for non-module model. |
| 37-54 | Sampling loop | Crops to context length, runs model, scales logits by temperature, applies top-k filter, softmaxes, samples, checks for `"\nUser:"`, appends token. | Context tensor and sampling settings. | Longer context tensor. | Repeated model calls. | Reads `prompt_len` before outer function assigns it; top-k not clamped; if `max_new_tokens` is tuple, `range()` fails. |
| 56 | Return full context | Returns generated sequence including prompt. | Final context. | Tensor `(B, original+new)`. | None. | Caller slices generated suffix. |
| 58-64 | Outer decode | Encodes prompt, calls nested generator, computes prompt length, decodes generated suffix, returns answer. | Prompt string. | Answer string. | None. | `prompt_len` is computed too late for nested stop check. |
| 66-73 | Script-level execution | Unpickles `model.pkl`, defines hard-coded prompt and settings, prints `generate_text(...)`. | `model.pkl`. | Printed text. | Reads pickle. | Not guarded by `if __name__ == "__main__"`; `max_new_tokens = 120,` creates tuple. |

**Potential interview talking points:**

- The intended sampling loop is standard autoregressive generation.
- The code crops context to `model.context_length`, which allows long generated sequences while respecting positional embedding limits.
- Temperature and top-k are implemented directly on logits before softmax.

**Possible improvements or risks:**

- Move `prompt_len` calculation before nested `generate()` is called.
- Remove trailing comma from `max_new_tokens = 120,`.
- Clamp `top_k` as done in `frontend/app.py`.
- Use EOS stopping consistently.
- Add CLI arguments for prompt and generation settings.
- Add `if __name__ == "__main__"` guard so importing `generate_text` does not load a pickle and execute generation.
- Avoid pickle for untrusted artifacts.

### `src/Multi_Headed_Attention.py`

**Role:** Combines multiple causal attention heads into a multi-head self-attention layer.

**Why it matters:** Multi-head attention is the core parallel attention mechanism in transformer blocks, allowing different heads to learn different attention patterns.

**Key dependencies/imports:**

- `torch`: concatenation.
- `torch.nn`: module, module list, linear projection.
- `torchtyping.TensorType`: annotations.
- `SingleHeadAttention`: per-head causal attention implementation.

**Exports/public surface:**

- `MultiHeadedSelfAttention`
- `MultiHeadedSelfAttention.__init__`
- `MultiHeadedSelfAttention.forward`

**Used by:** `src/Transformer_Block.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-4 | Imports | Loads tensor utilities, module classes, annotations, and single-head attention. | Installed packages and local module path. | Imported symbols. | Import failure if path missing. | Direct local import requires `src` on `sys.path`. |
| 6 | Class declaration | Declares PyTorch attention module. | PyTorch module system. | Module class. | None. | Used inside transformer block. |
| 8-11 | `__init__` | Creates `num_heads` independent `SingleHeadAttention` modules, each with dimension `attention_dim // num_heads`; creates output projection. | `embedding_dim`, `attention_dim`, `num_heads`. | Initialized attention module. | Allocates parameters. | No assertion that `attention_dim` is divisible by `num_heads`; if not divisible, concatenated head dim may be smaller than `attention_dim`, breaking `out_proj`. |
| 13-18 | `forward` | Runs each head on same embedded input, concatenates outputs along last dimension, applies output projection. | Embedded tensor `(B, T, D)`. | Projected tensor `(B, T, D)` when dimensions match. | None. | Python loop over heads is simple and readable but less optimized than fused QKV. |

**Potential interview talking points:**

- This implementation is deliberately explicit: each head is its own module with separate Q/K/V layers.
- The output projection mixes information across heads after concatenation.
- The loader script maps Hugging Face fused QKV weights into this per-head layout.

**Possible improvements or risks:**

- Add divisibility assertion.
- Rename `attention_dim` or ensure it always equals `embedding_dim` as used in `TransformerBlock`.
- Use a vectorized/fused implementation for performance.
- Add dropout on attention weights/output if matching GPT-2 more closely.

### `src/Self_Attention_Mechanism.py`

**Role:** Implements one causal self-attention head.

**Why it matters:** This is the lowest-level transformer mechanism: it computes Q, K, V, masks future tokens, normalizes attention scores, and returns weighted values.

**Key dependencies/imports:**

- `torch`: transpose, matmul, mask creation, softmax.
- `torch.nn`: module and linear layers.
- `torchtyping.TensorType`: annotations.

**Exports/public surface:**

- `SingleHeadAttention`
- `SingleHeadAttention.__init__`
- `SingleHeadAttention.forward`

**Used by:** `src/Multi_Headed_Attention.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-3 | Imports | Loads PyTorch, module classes, and annotations. | Installed packages. | Imported symbols. | Import failure if missing. | `TensorType` is annotation only here. |
| 5 | Class declaration | Declares one attention head. | PyTorch module system. | Module class. | None. | One instance is created per head. |
| 7-12 | `__init__` | Creates separate linear projections for key, query, and value. | `embedding_dim`, `attention_dim`. | Learnable Q/K/V projections. | Allocates parameters. | Biases are enabled, matching the loader's bias copying. Softmax module is commented out. |
| 14-17 | Project K/Q/V | Applies linear layers to embedded sequence. | Embedded tensor `(B, T, D)`. | `k`, `q`, `v` shaped `(B, T, head_dim)`. | None. | Assumes 3-D batch/time/channel input. |
| 18-19 | Score computation | Transposes keys and computes scaled dot-product scores with in-place division by sqrt(head_dim). | `q`, `k`. | Scores `(B, T, T)`. | In-place modifies `qkt` only. | Uses `k.shape[-1]**0.5`, the standard scale. |
| 20-22 | Old mask comments | Commented earlier mask/softmax approach. | None. | None. | None. | Documents iteration history. |
| 23-25 | Causal mask | Builds lower-triangular boolean mask and fills future positions with `-inf`. | Sequence length `T`, score device. | Masked scores. | Allocates mask each forward pass. | Mask shape `(T, T)` broadcasts over batch. |
| 26-28 | Softmax and weighted sum | Softmaxes masked scores and multiplies by values. | Masked scores, `v`. | Head output `(B, T, head_dim)`. | None. | No dropout on attention weights. |

**Potential interview talking points:**

- Causal masking is what makes the model decoder-only and autoregressive.
- The implementation uses separate per-head projections, making the math easier to inspect.
- The mask is device-aware, so it works on CPU or CUDA.

**Possible improvements or risks:**

- Cache or register the causal mask to avoid reallocating it every forward pass.
- Add attention dropout if matching GPT-2 training behavior.
- Use `torch.nn.functional.scaled_dot_product_attention` for speed in production-style code, though that would reduce educational transparency.
- Add shape assertions and tests for no future-token attention.

### `src/Training_Loop.py`

**Role:** Implements the scratch-training loop for next-token prediction.

**Why it matters:** It connects tokenized corpus windows to model optimization and checkpoint-producing training.

**Key dependencies/imports:**

- `torch`: device checks, optimizer, AMP, random permutations.
- `torch.nn`: gradient clipping utility.
- `torch.nn.functional`: cross-entropy loss.
- `make_tokens` from `Data_Preprocessing`: corpus tokenization.

**Exports/public surface:**

- `train_gpt(model, raw_text, batch_size, epochs=5, lr=3e-4, device="cuda", stride=None)`

**Used by:** `src/run_train.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-4 | Imports | Loads PyTorch, loss functions, and `make_tokens`. | Installed packages and local module. | Imported symbols. | Importing `Data_Preprocessing` reads `config.yaml`. | Direct import requires `src` on path. |
| 6-8 | Function signature and device setup | Selects CUDA only if requested and available; moves model and sets train mode. | Model, raw text, batch size, epochs, lr, device, stride. | Model on selected device in train mode. | Mutates model device/mode. | Any non-`"cuda"` requested device becomes CPU. |
| 10-14 | Tokenize once | Converts raw text to tokens, ensures tensor, moves token tensor to device. | Raw text. | Token tensor on device. | GPU memory use if CUDA. | Moving full corpus to GPU can be fast but memory-heavy for larger corpora. |
| 16-23 | Window creation and validation | Reads model context length, defaults stride to context length, unfolds tokens into `(Nw, T+1)` windows, raises if none. | Token tensor, context length, stride. | Window view and count. | None. | Non-overlapping windows by default; stride 1 would create more overlapping examples. |
| 25-29 | Optimizer setup | Tries fused AdamW on CUDA; falls back if PyTorch version lacks fused argument. | Model params, lr. | AdamW optimizer. | Allocates optimizer state during training. | Weight decay fixed at 0.1. |
| 31-32 | AMP scaler setup | Enables AMP only on CUDA. | Device. | GradScaler. | None. | Uses `torch.amp.GradScaler('cuda', enabled=use_amp)`. |
| 34-39 | Epoch and batch loops | Shuffles window indices per epoch, initializes loss counters, iterates batches. | `epochs`, `Nw`, `batch_size`. | Batch index slices. | Randomness from `torch.randperm`. | Last batch may be smaller than `batch_size`. |
| 40-44 | Split windows | Selects batch windows and splits into input `x` and target `y`. | Window indices. | `(B, T)` input and target. | None. | Standard next-token shift. |
| 46-53 | Forward and loss | Clears grads, runs autocast, computes logits, flattens logits/targets, computes cross-entropy. | Model, x, y. | Scalar loss. | None. | No validation loss or metrics beyond train loss. |
| 55-59 | Backward and step | Scales loss, backprops, unscales, clips grad norm to `1.0`, optimizer step, scaler update. | Loss, model params. | Updated model weights. | Mutates model/optimizer state. | Gradient clipping helps stability. |
| 61-64 | Loss logging | Accumulates loss and prints average per epoch. | Batch losses. | Console output. | Writes stdout. | If `steps` somehow zero, divide by zero; `Nw > 0` prevents this. |

**Potential interview talking points:**

- Tokenization is done once rather than per batch.
- `unfold` creates efficient training windows.
- AMP and fused AdamW are used opportunistically on CUDA.
- Gradient clipping is included for training stability.

**Possible improvements or risks:**

- Add validation split and perplexity reporting.
- Save checkpoints during training, not only at the end.
- Save optimizer/scaler state for resume.
- Make weight decay, batch size, epochs, stride, and model size configurable.
- Avoid moving very large corpora fully onto GPU if scaling up.
- Add deterministic seeding controls.

### `src/Transformer_Block.py`

**Role:** Defines a pre-layernorm transformer block and its feed-forward submodule.

**Why it matters:** This file composes attention, residual connections, layer normalization, and MLP logic into the repeated unit of the GPT model.

**Key dependencies/imports:**

- `torch`: imported but not directly used.
- `torch.nn`: module, layer norm, linear, GELU, dropout.
- `torchtyping.TensorType`: annotations.
- `MultiHeadedSelfAttention`: attention sublayer.

**Exports/public surface:**

- `TransformerBlock`
- `TransformerBlock.FeedForward`

**Used by:** `src/GPT.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-4 | Imports | Loads PyTorch modules, annotations, and multi-head attention. | Installed packages and local module path. | Imported symbols. | Import failure if path missing. | `torch` itself is unused. |
| 6 | Class declaration | Declares transformer block. | PyTorch module system. | Module class. | None. | Repeated in `GPT.nxTransformer`. |
| 8-13 | `__init__` | Creates first layer norm, multi-head attention, second layer norm, and feed-forward submodule. | `model_dim`, `num_heads`. | Initialized block. | Allocates parameters. | Uses pre-layernorm architecture. |
| 15-20 | `forward` | Applies norm -> attention -> residual -> norm -> feed-forward -> residual. | Embedded sequence `(B, T, D)`. | Updated sequence `(B, T, D)`. | None. | Residual shapes must match; attention and feed-forward both return `D`. |
| 22-29 | `FeedForward.__init__` | Creates MLP with up projection `D -> 4D`, GELU approximate tanh, down projection `4D -> D`, dropout p=0.2. | `model_dim`. | Initialized MLP. | Allocates parameters. | Dropout is active in train mode and disabled in eval mode. |
| 31-32 | `FeedForward.forward` | Applies up projection, GELU, down projection, and dropout. | Tensor `(B, T, D)`. | Tensor `(B, T, D)`. | Dropout randomness in train mode. | No separate residual here; outer block adds residual. |

**Potential interview talking points:**

- Pre-layernorm improves optimization stability in deeper transformer stacks.
- The MLP expansion ratio `4D` follows common GPT-style transformer design.
- Residual connections preserve information and gradient flow.

**Possible improvements or risks:**

- Make dropout probability configurable.
- Remove unused `torch` import.
- Add tests for output shape preservation.
- Consider whether dropout should match DistilGPT2 when validating weights; dropout is disabled in eval mode for validation and generation.

### `src/load_pretrained_model.py`

**Role:** Loads Hugging Face DistilGPT2 weights into the custom GPT architecture and validates output similarity.

**Why it matters:** It is the most technically distinctive file. It proves the custom architecture is close enough to receive real pretrained weights and requires understanding of Hugging Face GPT-2 parameter layouts.

**Key dependencies/imports:**

- `torch`: tensor ops, device checks, no-grad, copy operations.
- `GPT`: custom model architecture.
- `pathlib.Path`: generated artifact deletion.
- `pickle`: save transferred model.
- `transformers.AutoModelForCausalLM`: load DistilGPT2 model.
- `transformers.AutoTokenizer`: tokenize validation prompt.

**Exports/public surface:**

- `load_distilgpt2_into_my_gpt(my_model, device="cpu")`
- Script-level `my`, `model`, `hf`, and validation/save behavior.

**Used by:** Developer command `python src/load_pretrained_model.py`. README says run this before launching Streamlit.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-4 | Imports | Loads PyTorch, custom GPT, Path, and pickle. | Installed packages and local modules. | Imported symbols. | Import failure if path missing. | Transformers imports are delayed/partly later. |
| 6-12 | DistilGPT2-shaped `my` | Creates custom model with `vocab_size=50257`, `context_length=1024`, `model_dim=768`, `num_blocks=6`, `num_heads=12`. | Hard-coded dimensions. | Custom model instance. | Allocates large model. | Matches DistilGPT2 dimensions by repo evidence. |
| 14-20 | Loader function and docstring | Defines conversion purpose: HF Conv1D and fused QKV to local linears. | Custom model. | Function. | None. | Good inline documentation of core challenge. |
| 20-25 | Load Hugging Face model | Imports and loads `AutoModelForCausalLM.from_pretrained("distilgpt2")`, moves to device, evals, grabs state dict. | `device`, internet/cache. | HF model and state dict. | May download model. | Network/cached model requirement. |
| 27-31 | Shape setup | Gets custom model width `D`, number of heads `H`, asserts divisibility, computes `hd`. | Custom model. | Head dimension. | Assertion may raise. | Assumes at least one block exists. |
| 33-38 | Embeddings copy | Copies token embeddings and sliced positional embeddings. | HF state dict. | Custom embedding weights. | Mutates custom model parameters. | Allows custom context length <= HF positional length. |
| 40-49 | Block loop and LayerNorms | Iterates custom transformer blocks and copies `ln_1`/`ln_2` weights and biases. | HF state dict block keys. | Custom norm params. | Mutates parameters. | Number of custom blocks must match available HF layers. |
| 51-57 | Fused QKV split | Reads HF `attn.c_attn.weight` and bias, splits into Q/K/V matrices and biases. | HF fused attention params. | `Wq`, `Wk`, `Wv`, `bq`, `bk`, `bv`. | None. | HF GPT-2 Conv1D stores weight as `(in, out)`. |
| 58-72 | Per-head Q/K/V copy | For each head, slices Q/K/V columns, transposes to `(out, in)`, copies weights and biases into local head linears. | Split Q/K/V, head index. | Head-specific local weights. | Mutates per-head modules. | This is the core layout translation. |
| 74-79 | Attention output projection | Copies HF `c_proj` into local `out_proj`, transposing weight. | HF projection params. | Local attention output projection. | Mutates parameters. | Handles Conv1D-style layout. |
| 81-92 | MLP copy | Copies HF `mlp.c_fc` and `mlp.c_proj` into local feed-forward projections, transposing weights. | HF MLP params. | Local MLP weights/biases. | Mutates parameters. | Shape must match `D -> 4D -> D`. |
| 94-97 | Final LayerNorm | Copies final transformer layer norm. | HF `ln_f` params. | Custom final norm. | Mutates parameters. | Direct shape match required. |
| 99-104 | LM head | Copies HF language model head weight and zeroes local bias. | HF `lm_head.weight`. | Custom output projection. | Mutates parameters. | HF head has no bias; local `nn.Linear` does. |
| 106-108 | Return models | Moves custom model to device, evals, returns custom and HF model. | Custom model and HF model. | `(my_model, hf)`. | Mutates mode/device. | Caller can compare outputs. |
| 110-117 | Tokenizer/device/script setup | Imports tokenizer, selects device, runs loader, tokenizes validation prompt. | CUDA availability, prompt text. | `model`, `hf`, `x`. | May download tokenizer. | No main guard; executing import runs all script behavior. |
| 119-125 | Validation prints | Runs both models, prints max abs logit diff and final-position argmax IDs. | Token IDs and models. | Console diagnostics. | Inference compute. | Sanity check, not a full equivalence test. |
| 128-132 | Save pickle | Deletes existing `pre_trained_model.pkl` and pickles custom model. | Custom model. | `pre_trained_model.pkl`. | Deletes/replaces file. | Pickle artifact is configured for LFS if tracked. |

**Potential interview talking points:**

- The file demonstrates understanding of GPT-2's fused QKV layout.
- It handles transposition from Hugging Face Conv1D layout to PyTorch `nn.Linear`.
- It validates transferred weights by comparing logits and argmax token IDs.
- It bridges educational model code with a real pretrained checkpoint.

**Possible improvements or risks:**

- Add `if __name__ == "__main__"` guard.
- Save a state dict plus config instead of pickling whole model.
- Add automated tests comparing selected layers and logits within tolerance.
- Parameterize model name and output path.
- Handle offline/cached model errors gracefully.
- Document expected max absolute difference range.
- Consider tied embeddings/head behavior and dropout implications if chasing exact parity.

### `src/run_train.py`

**Role:** Scratch-training script that builds a smaller custom GPT model, trains it on `train.txt`, and saves artifacts.

**Why it matters:** It is the main end-to-end training entrypoint for the from-scratch path.

**Key dependencies/imports:**

- `torch`: CUDA detection and state dict saving.
- `tiktoken`: vocab size lookup.
- `pickle`: whole-model serialization.
- `GPT`: model class.
- `train_gpt`: training loop.
- `yaml`: config parsing.
- `time`: elapsed-time measurement.
- `pathlib.Path`: deleting existing artifacts.

**Exports/public surface:**

- `load_hyperparameters`
- `read_text`
- `main`
- Script behavior under `if __name__ == "__main__"`.

**Used by:** Developer command `python src/run_train.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-8 | Imports | Loads training, tokenizer, model, config, time, and filesystem dependencies. | Installed packages and local modules. | Imported symbols. | Importing `Training_Loop` imports `Data_Preprocessing`, which reads `config.yaml`. | Direct imports require script execution/path context. |
| 10-14 | `load_hyperparameters` | Reads YAML config using `safe_load`. | YAML path. | Config dictionary. | Reads file. | Duplicates helper from other files. |
| 16-18 | Module-level config load | Loads `config.yaml` and stores `vocab`. | Root-relative config. | Tokenizer name. | Reads file at import time. | Cwd-sensitive. |
| 20-22 | `read_text` | Opens text file with UTF-8 and replacement errors. | Path string. | Raw text string. | Reads file. | `errors="replace"` avoids decode crashes at cost of replacement characters. |
| 24-31 | `main` setup | Selects device, reads `train.txt`, creates tiktoken encoder, gets vocab size. | CUDA availability, corpus, config. | `device`, `raw_text`, `vocab_size`. | Reads corpus. | `train.txt` is large but manageable in memory here. |
| 33-40 | Model construction | Builds `GPT` with vocab size, context 256, dimension 512, 12 blocks, 8 heads. | Hard-coded hyperparameters. | Model instance. | Allocates parameters. | Smaller than DistilGPT2 in hidden size/context, but same block count count is 12 vs DistilGPT2's 6 in loader. |
| 42-54 | Training call and timing | Calls `train_gpt` with batch size 32, epochs 10, lr 3e-4, stride=context length; prints elapsed time. | Model, text, hyperparameters. | Trained model and stdout. | Mutates model weights. | Non-overlapping windows by default. |
| 56-60 | Delete old artifacts | Deletes `model.pkl` and `gpt_weights.pt` if present. | Paths. | Clean target paths. | Removes files. | Destructive for prior generated artifacts; `missing_ok=True` avoids errors if absent. |
| 62-65 | Save artifacts | Saves state dict to `gpt_weights.pt`, pickles whole model to `model.pkl`, prints saved message. | Trained model. | Files on disk. | Writes files. | Print only mentions `gpt_weights.pt`, not `model.pkl`. |
| 67-68 | Main guard | Runs `main()` only when script is executed directly. | Python execution context. | Training run. | See above. | Good import hygiene compared with `Generator.py` and `load_pretrained_model.py`. |

**Potential interview talking points:**

- The script ties tokenizer vocab size to model output dimension.
- It trains a smaller demonstration model than the pretrained DistilGPT2-shaped model.
- It saves both a state dict and a full pickle, illustrating two serialization styles.

**Possible improvements or risks:**

- Make hyperparameters configurable.
- Add checkpoint metadata and optimizer state.
- Log both artifacts saved.
- Add train/validation split.
- Avoid deleting artifacts until after new saves succeed, to reduce risk of losing previous checkpoints on failure.
- Add dependency and path checks.

### `train.txt`

**Role:** Large local text corpus used for scratch training.

**Why it matters:** It is the source data for `src/run_train.py` and `src/Training_Loop.py`. Without it, the scratch-training workflow cannot run as written.

**Key dependencies/imports:** Read by Python file I/O; tokenized by tiktoken.

**Exports/public surface:** Corpus text.

**Used by:** `src/run_train.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| Whole file | Large plain-text corpus, 10,827,302 bytes, 37,333 lines observed, tracked via Git LFS | Provides local training text for next-token prediction. | Text file. | Raw text loaded by `read_text("train.txt")`. | None when read. | Detailed line-by-line decoding is not useful for code analysis. Sample content appears to be encyclopedia-like article text with tokenization artifacts such as `<unk>` and `@-@`. |

**Potential interview talking points:**

- The project includes a local corpus so the training path is self-contained once dependencies are installed.
- Because this is demo-scale local training, output quality should not be overclaimed.

**Possible improvements or risks:**

- Document corpus provenance/licensing if needed; unclear from repo evidence.
- Add train/validation split.
- Add data preprocessing notes and expected token count.
- Keep LFS working; current `git status --short` produced LFS clean-filter warnings on this working copy.

## 11. Cross-Cutting Concerns

### Security and secrets handling

- No `.env` file is tracked.
- `.gitignore` ignores `.env`, `.envrc`, and virtual environment folders.
- No API keys, tokens, or credentials were found in tracked source/config files.
- `pickle.load()` is used in `src/Generator.py` and `frontend/app.py`. This is safe only for trusted local artifacts. It is not safe for arbitrary user-uploaded files.
- `yaml.safe_load()` is used, which is the safer YAML parsing choice.
- No authentication or authorization exists because the repo is not an API service.

### Error handling

- Streamlit startup errors are shown to users with `st.error()` and `st.stop()`.
- Training detects no-window data and raises `ValueError`.
- Weight transfer uses an assertion for `model_dim % num_heads`.
- Most scripts do not catch dependency, file, CUDA memory, Hugging Face download, or generation-time errors.

### Logging and observability

- Training prints average epoch loss.
- Pretrained loader prints max absolute logit difference and argmax IDs.
- No structured logs, metrics files, experiment tracking, dashboards, or monitoring config.
- No loss curves are saved.

### Testing strategy

- No tests are tracked.
- Validation is currently manual/scripted through `src/load_pretrained_model.py`.
- There are no automated checks for tensor shapes, causal masks, data shifting, generation, UI helpers, or checkpoint loading.

### Performance considerations

- Attention mask is recreated every `SingleHeadAttention.forward()` call.
- Attention is implemented as Python-looped per-head modules rather than fused batched projection.
- `train_gpt()` tokenizes once and uses `unfold`, which is efficient for the educational scope.
- Moving all tokens to GPU can speed training but may not scale to larger corpora.
- Streamlit loads the model at script run; there is no `st.cache_resource`, so reload behavior depends on Streamlit reruns and session behavior.

### Scalability considerations

- No distributed training.
- No batching for multiple UI users beyond Streamlit's process behavior.
- No model serving API, queue, or concurrency design.
- No checkpoint sharding.
- No config management for larger experiments.

### Accessibility

- Streamlit provides baseline UI widgets.
- The app includes sidebar sliders and chat input.
- No custom accessibility testing, labels beyond widget labels, or keyboard-flow documentation is present.
- Screenshot has no detailed alt text beyond README image label.

### Data privacy

- User inputs in Streamlit are kept in `st.session_state.messages` for the session.
- No database persistence is implemented.
- No external API call is made during generation itself; Hugging Face calls happen during pretrained loading.
- No privacy notice exists in the repo.

### Dependency management

- Dependencies are unpinned.
- `streamlit` and `PyYAML` are missing from `requirements.txt`.
- No lockfile.
- No Python version specified.

### Code organization

- Clear conceptual split between model internals, training, generation, pretrained loading, and frontend.
- Local imports are direct module imports rather than package-relative imports.
- Several helpers are duplicated (`load_hyperparameters`).
- Generated artifact paths are hard-coded at repo root.

### Maintainability

Strengths:

- Small codebase.
- Explicit transformer internals.
- README explains intent and limitations.

Risks:

- No tests.
- Hard-coded paths and hyperparameters.
- Pickle-based model storage.
- Missing dependency declarations.
- Script-level side effects in `Generator.py` and `load_pretrained_model.py`.

### Deployment readiness

- README links a Streamlit deployment.
- No tracked deployment configuration.
- No production incident playbook.
- No Dockerfile or environment lock.
- The UI expects a local `pre_trained_model.pkl`, but that artifact is not tracked.

### Failure modes

- Missing dependencies.
- Missing generated checkpoints.
- Hugging Face download failure.
- CUDA out-of-memory.
- Long prompt exceeding context if not cropped; Streamlit and generator crop, direct `GPT.forward()` does not.
- Untrusted pickle execution risk.
- LFS issues for large files.
- Standalone generator runtime bugs.

### Technical debt

- Incomplete dependency manifest.
- No tests/CI.
- Repeated config loader.
- Hard-coded runtime paths.
- No structured config for model/training/generation settings.
- Pickle artifacts instead of safer state dict plus config loading.

## 12. Testing And Validation

### Test frameworks used

No test framework is used in tracked files. There is no `pytest`, `unittest`, `tests/`, or CI configuration.

### Tests that exist

No automated tests exist.

### Validation that exists

`src/load_pretrained_model.py` performs a manual sanity check:

- Load Hugging Face DistilGPT2.
- Load its weights into the custom architecture.
- Tokenize `"Hello, I am a language model"`.
- Compute Hugging Face logits and custom logits.
- Print maximum absolute difference.
- Print final-position argmax token IDs for both models.

This is useful, but it is not a test suite. It is a runtime script with external dependency and generated artifact side effects.

### Behavior covered by current validation

- Some level of forward-pass compatibility after weight transfer.
- Attention/MLP/norm/head weight shapes are compatible enough for `copy_`.
- Custom model can run on the same token input as DistilGPT2.

### Behavior untested

- `SingleHeadAttention` causal mask correctness.
- Multi-head output shape and head concatenation.
- `TransformerBlock` residual and feed-forward shape preservation.
- `GPT.forward()` output shape and context-length failure behavior.
- Token shifting in `batch_loader`, `make_tokens`, and `batch_loader_stride_tokens`.
- Training loop on a tiny corpus.
- Generator stopping, top-k, temperature, seed determinism, and prompt slicing.
- Streamlit helper functions.
- Config loading failure paths.
- Artifact save/load round trips.
- CWD/path assumptions.
- Dependency manifest completeness.

### How to run validation manually

From repo root:

```bash
python src/load_pretrained_model.py
```

From repo root, after generating `pre_trained_model.pkl`:

```bash
streamlit run frontend/app.py
```

From repo root, after generating `model.pkl`:

```bash
python src/Generator.py
```

Current caution: `src/Generator.py` appears to contain runtime issues described above.

### Suggested high-value tests to add

| Test | Purpose | Files |
|---|---|---|
| Tiny tokenizer/window test | Verify `x` and `y` are shifted by one token. | `src/Data_Preprocessing.py` |
| Causal mask test | Verify token position `t` cannot attend to positions `> t`. | `src/Self_Attention_Mechanism.py` |
| Shape tests | Verify attention, block, and GPT output shapes for small dimensions. | Model files |
| Divisibility test | Verify model construction fails clearly when `model_dim % num_heads != 0`. | `src/GPT.py`, `src/Multi_Headed_Attention.py` |
| Tiny training smoke test | Train for one epoch on tiny text and assert loss is finite. | `src/Training_Loop.py` |
| Generation deterministic seed test | Same seed should produce same output for a fixed small mock model. | `src/Generator.py`, `frontend/app.py` |
| Streamlit prompt builder test | Verify user/assistant messages format as expected. | `frontend/app.py` |
| Pretrained loader unit mapping test | Verify Q/K/V slicing/transposition on synthetic tensors. | `src/load_pretrained_model.py` |
| Dependency/import test | Verify all tracked modules import in expected working directory. | Whole repo |

## 13. Build, Deployment, And Operations

### Build process

No build process is tracked. The project is run directly as Python scripts and a Streamlit app.

### Runtime process

Scratch training:

```bash
python src/run_train.py
```

Pretrained weight transfer:

```bash
python src/load_pretrained_model.py
```

Streamlit runtime:

```bash
streamlit run frontend/app.py
```

### Deployment clues

- README links a Streamlit-hosted demo.
- `frontend/app.py` is compatible with Streamlit's execution model.
- No `.streamlit/`, `packages.txt`, Dockerfile, GitHub Actions, Procfile, or cloud config is tracked.
- `pre_trained_model.pkl` is required by the app but not tracked in the observed file inventory.

### Docker/Kubernetes/cloud config

None tracked.

### CI/CD config

None tracked.

### Monitoring/logging config

None tracked.

### Operational risks

- Missing dependencies in `requirements.txt` may break deploys.
- Missing `pre_trained_model.pkl` may break Streamlit startup.
- Pickle model artifact may be too large for some deploy targets and unsafe if not trusted.
- Hugging Face download may fail in restricted environments.
- No health checks or startup tests.
- No automated tests to catch regressions before deploy.
- No pinned dependencies means upstream changes can alter behavior.

### How a production incident might be debugged from this codebase

Although the repo is not production-oriented, practical debugging would start with:

1. Startup fails with config error: inspect `frontend/app.py:29-33` and verify `config.yaml` exists relative to process cwd.
2. Startup fails with tokenizer error: inspect `config.yaml` and tiktoken installation.
3. Startup fails with model load error: verify `pre_trained_model.pkl` exists and was generated by `src/load_pretrained_model.py`.
4. Generation fails: inspect `frontend/app.py:74-126`, model device, model context length, and top-k/temperature values.
5. Bad outputs: confirm whether model is scratch-trained `model.pkl` or pretrained-transferred `pre_trained_model.pkl`; verify prompt construction in `build_prompt()`.
6. Training fails: inspect corpus length, CUDA memory, and `src/Training_Loop.py`.
7. Pretrained transfer mismatches: inspect printed max absolute difference and mapping logic in `src/load_pretrained_model.py`.

## 14. How To Modify Or Extend This Project

### How to add a new feature

Follow the existing separation:

- Model architecture changes belong in `src/GPT.py`, `src/Transformer_Block.py`, `src/Multi_Headed_Attention.py`, or `src/Self_Attention_Mechanism.py`.
- Training behavior belongs in `src/Training_Loop.py` or `src/run_train.py`.
- Sampling behavior belongs in `src/Generator.py` and `frontend/app.py`.
- UI controls belong in `frontend/app.py`.
- Tokenizer/global config currently belongs in `config.yaml`, but broader config would require code changes.

Because the pretrained loader depends on exact architecture names and shapes, any model architecture change may require corresponding updates to `src/load_pretrained_model.py`.

### How to add a new route/page/endpoint/command

There are no web routes or API endpoints. To add a new command, follow `src/run_train.py` style:

- Put logic behind a `main()` function.
- Use `if __name__ == "__main__": main()`.
- Read files from repo root or resolve paths explicitly.
- Avoid import-time side effects.

For Streamlit, add UI controls in `frontend/app.py`, then pass values into `generate_text()` or other helpers.

### How to add a new data model or config

Current config is minimal. A practical extension would be:

- Add keys to `config.yaml`, such as model sizes, artifact paths, training defaults, generation defaults.
- Update config loaders to validate required keys.
- Replace duplicate `load_hyperparameters()` functions with one shared helper.
- Avoid loading config at import time.

### How to add tests

Recommended path:

1. Add `pytest` to a dev requirements file or `requirements.txt`.
2. Create `tests/`.
3. Start with fast CPU tests using tiny dimensions and tiny token sequences.
4. Test pure functions/helpers before training-heavy flows.
5. Use synthetic tensors to test weight slicing in `load_pretrained_model.py`.
6. Keep Hugging Face download tests optional or marked slow.

### How to debug common issues

- Import issues: run scripts from repo root; ensure `src` path is available.
- Tokenizer issues: inspect `config.yaml` and tiktoken install.
- Shape issues: print `context.shape`, `logits.shape`, `model.context_length`, `model_dim`, and `num_heads`.
- Training issues: lower batch size, context length, or model dimension if memory fails.
- Streamlit issues: verify `pre_trained_model.pkl` exists and dependencies include Streamlit/PyYAML.
- Generation quality: distinguish scratch-trained local model from DistilGPT2-transferred model.

### How to avoid breaking existing patterns

- Preserve raw logits from `GPT.forward()`; apply softmax only in loss/sampling.
- Preserve causal masking in self-attention.
- Preserve shifted next-token target construction.
- Keep tokenizer and vocab size aligned.
- If changing model dimensions, update training scripts and pretrained loader assumptions together.
- Do not silently change artifact paths without updating README and UI.

## 15. Interview Preparation Pack

### 15.1 Elevator Pitches

30-second pitch:

"This is a from-scratch GPT-2-style decoder-only transformer in PyTorch. It implements causal self-attention, multi-head attention, transformer blocks, next-token training, autoregressive sampling, and a Streamlit demo. The standout part is a DistilGPT2 weight-transfer script that maps Hugging Face's fused attention weights into my custom per-head modules and validates the custom model against reference logits."

60-second pitch:

"I built an educational GPT-2-style transformer to show that I understand what happens below the model API. The repo manually implements token and positional embeddings, pre-layernorm transformer blocks, multi-head causal self-attention, a feed-forward network, and a training loop for next-token prediction on a local text corpus. It also includes token-by-token generation with temperature and top-k sampling, plus a Streamlit interface. The most technical piece is the DistilGPT2 loader: Hugging Face stores GPT-2 attention projections as fused QKV Conv1D-style weights, while my implementation stores separate query, key, and value linears per head. I split, slice, transpose, and copy those weights into my architecture, then compare logits against the Hugging Face model."

2-minute technical pitch:

"The project is a compact decoder-only transformer implementation. The data path starts with `train.txt`, which is tokenized using tiktoken's GPT-2 encoding from `config.yaml`. The training loop unfolds the token stream into windows of length `context_length + 1`, uses the first `T` tokens as input and the next `T` tokens as targets, then optimizes cross-entropy over flattened `(B*T, V)` logits. The model itself has learned token embeddings and positional embeddings, then a stack of pre-layernorm transformer blocks. Each block normalizes, runs multi-head causal self-attention, adds a residual, normalizes again, runs a `D -> 4D -> D` GELU feed-forward network, and adds a second residual. Attention is implemented explicitly: Q, K, and V are separate linears, scores are scaled by square root head dimension, a lower-triangular mask blocks future positions, and softmaxed weights multiply V. For inference, generation crops context to the model context length, takes final-token logits, applies temperature and optional top-k, softmaxes, samples, and appends one token at a time. The DistilGPT2 weight loader is the bridge to a real pretrained model: it constructs a custom model with DistilGPT2 dimensions, loads Hugging Face weights, splits fused QKV parameters, slices per head, transposes from Conv1D layout to `nn.Linear`, copies MLP/norm/head weights, and validates output logits."

Recruiter-friendly pitch:

"I built a hands-on GPT-2 reconstruction in PyTorch to demonstrate real understanding of language model internals. It includes training, generation, pretrained weight loading, and a small web demo. It is a strong portfolio project because it goes beyond calling an API and shows I can reason about model architecture, tensors, and deployment-facing demos."

Senior-engineer technical pitch:

"This repository is intentionally small but goes deep on architecture fidelity. The core modules implement a decoder-only transformer without leaning on `nn.Transformer`: explicit per-head Q/K/V projections, causal masking, pre-LN residual blocks, and raw-logit outputs. The training path uses tokenized sliding windows, cross-entropy, AdamW, optional CUDA AMP, and grad clipping. The most interesting engineering work is reconciling two parameter layouts: Hugging Face GPT-2 uses fused QKV Conv1D-style weights, while this implementation uses independent per-head `nn.Linear` projections. The loader performs deterministic slicing/transposition and validates logits against DistilGPT2. The main production gaps are known: no tests, incomplete dependency manifest, pickle artifacts, hard-coded config, and no CI."

### 15.2 Architecture Questions And Answers

**Q: Why did this project use a decoder-only transformer architecture?**

A: The repository is modeling GPT-2-style next-token prediction. Decoder-only transformers are designed for autoregressive generation because each position can attend only to previous positions and itself. That behavior is implemented in `src/Self_Attention_Mechanism.py` with a lower-triangular causal mask.

**Q: What are the main components of the model?**

A: `src/GPT.py` defines token embeddings, positional embeddings, a stack of `TransformerBlock`s, final layer norm, and vocabulary projection. `TransformerBlock` contains pre-layernorm multi-head self-attention and a feed-forward network. `MultiHeadedSelfAttention` wraps multiple `SingleHeadAttention` modules. `SingleHeadAttention` performs Q/K/V projection, scaled dot-product attention, causal masking, softmax, and value aggregation.

**Q: How does data flow through training?**

A: `src/run_train.py` reads `train.txt`, gets GPT-2 vocab size from tiktoken, builds a `GPT` model, and calls `train_gpt()`. `train_gpt()` tokenizes text once through `make_tokens()`, unfolds tokens into windows of `T + 1`, splits each window into `x` and shifted `y`, computes logits, flattens logits and targets, applies cross-entropy, and updates model parameters with AdamW.

**Q: Why does `GPT.forward()` return logits instead of probabilities?**

A: Raw logits are the correct interface for `F.cross_entropy`, which internally applies log-softmax. They also allow generation code to apply temperature, top-k filtering, or other sampling transforms before softmax.

**Q: How is causal masking implemented?**

A: `SingleHeadAttention.forward()` creates a lower-triangular boolean mask of shape `(T, T)`, fills positions outside the mask with `-inf`, and then softmaxes over the last dimension. Future positions therefore receive zero probability after softmax.

**Q: What makes the DistilGPT2 loader hard?**

A: Hugging Face GPT-2 attention stores Q, K, and V in one fused `c_attn` tensor and uses Conv1D-style weight layout. This repo stores Q, K, and V as separate `nn.Linear` layers inside each attention head. `src/load_pretrained_model.py` must split Q/K/V, slice each per head, transpose into linear layout, and copy the weights and biases.

**Q: How does the Streamlit app communicate with the model?**

A: There is no backend API. `frontend/app.py` imports the model classes by adding `src/` to `sys.path`, unpickles `pre_trained_model.pkl`, encodes the prompt with tiktoken, calls the model directly, samples tokens, and renders output in Streamlit.

**Q: Where are the bottlenecks?**

A: The explicit per-head attention implementation loops over heads and creates masks on every forward pass. Training moves the full token tensor to GPU, which can become memory-heavy for larger corpora. Streamlit loads a full pickled model into process memory.

**Q: What would fail first when scaling this?**

A: Likely memory and speed. Attention is quadratic in context length, full-corpus GPU loading does not scale to huge datasets, and Python-looped per-head attention is slower than fused attention. Operationally, missing dependency pins and pickle artifact handling would also fail in reproducible deployment.

**Q: How would you deploy it?**

A: Based on repo evidence, Streamlit is the intended deployment target. A deploy would need `requirements.txt` fixed to include Streamlit and PyYAML, a generated or bundled `pre_trained_model.pkl`, and a clear startup command `streamlit run frontend/app.py`. For a stronger deployment, replace pickle with state dict plus config and add health checks.

**Q: How would you monitor it?**

A: The repo has no monitoring. For a production-like extension, log startup model load time, device, generation latency, token counts, exceptions, and memory usage. Streamlit-level logs could capture generation errors. Training could log loss to a CSV or experiment tracker.

**Q: Why use pre-layernorm?**

A: `TransformerBlock.forward()` normalizes before attention and before feed-forward. Pre-layernorm is common in stable transformer training because residual paths remain direct and gradients flow more easily through deeper stacks.

**Q: Why does the training script use a smaller model than DistilGPT2?**

A: `src/run_train.py` uses context length 256 and model dimension 512, likely for demo/local training. `src/load_pretrained_model.py` uses DistilGPT2-compatible dimensions because exact shape compatibility is required for pretrained weight copying.

### 15.3 Code-Level Questions And Answers

**Q: What does `src/GPT.py:22` do?**

A: It creates position indices from `0` to `T-1` on the same device as the input context and unsqueezes them to shape `(1, T)`, allowing positional embeddings to broadcast across the batch.

**Q: Why is `self.linear` in `GPT` not followed by softmax?**

A: The model returns logits. Training uses `F.cross_entropy`, and generation applies sampling transforms before softmax. Adding softmax inside the model would make training less numerically stable and less flexible.

**Q: What does `tokens.unfold(0, T + 1, stride)` do in `Training_Loop.py`?**

A: It creates a view of contiguous token windows. Each row has `T + 1` tokens so the code can split it into `T` input tokens and `T` next-token targets.

**Q: What is the purpose of `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`?**

A: It limits gradient norm to stabilize training and reduce exploding-gradient risk.

**Q: Why does `load_pretrained_model.py` transpose weights before copying them?**

A: Hugging Face GPT-2 Conv1D weights are stored as `(in, out)`, while PyTorch `nn.Linear.weight` expects `(out, in)`. The transpose reconciles this layout difference.

**Q: How does `frontend/app.py` stop generation?**

A: It stops if the sampled token equals `eos_id` or if `max_new_tokens` iterations complete. It does not use a stop string.

**Q: How does `src/Generator.py` try to stop generation?**

A: It decodes text so far and breaks if `"\nUser:"` appears after the prompt. However, `prompt_len` is referenced before assignment, so the current implementation likely fails before that logic works.

**Q: What does `top_k_ui == 0` mean in Streamlit?**

A: The UI maps it to `top_k = None`, disabling top-k filtering.

**Q: Why does `frontend/app.py` import `GPT` even though it does not instantiate it?**

A: It likely ensures the class is available when unpickling a saved model object. Pickle needs the original class/module import path to reconstruct objects.

**Q: What is the side effect of `Data_Preprocessing.py` at import time?**

A: It immediately reads `config.yaml` and stores `vocab`. This makes imports cwd-sensitive.

**Q: What is risky about `run_train.py` deleting artifacts before saving?**

A: If saving fails after deletion, the previous `model.pkl` and `gpt_weights.pt` are gone. A safer approach is to write new files to temporary paths and replace after success.

**Q: What does `block.feed_forward.up_projection.weight.copy_(Wfc.T)` accomplish?**

A: It copies Hugging Face MLP up-projection weights into the local `nn.Linear(D, 4D)` layer after transposing from HF Conv1D layout to PyTorch linear layout.

### 15.4 Debugging Questions And Answers

**Scenario: Streamlit says `Model load error: Missing pre_trained_model.pkl`.**

- Symptom: App stops during startup with a model load error.
- Likely cause: `src/load_pretrained_model.py` has not been run, or it wrote the file somewhere else.
- Files to inspect: `frontend/app.py:42-53`, `frontend/app.py:166-170`, `src/load_pretrained_model.py:128-132`.
- How to reproduce: Run `streamlit run frontend/app.py` in a fresh clone without generating `pre_trained_model.pkl`.
- How to fix: Run `python src/load_pretrained_model.py` from repo root, then run Streamlit from repo root.
- How to prevent recurrence: Document artifact requirements and add startup checks/deploy scripts.

**Scenario: `ModuleNotFoundError: No module named 'streamlit'`.**

- Symptom: Streamlit command or import fails.
- Likely cause: `requirements.txt` does not include `streamlit`.
- Files to inspect: `requirements.txt`, `frontend/app.py:17`.
- How to reproduce: Install only `pip install -r requirements.txt`, then run Streamlit.
- How to fix: Install `streamlit`; update `requirements.txt`.
- How to prevent recurrence: Pin complete dependencies and add import smoke test.

**Scenario: `ModuleNotFoundError: No module named 'yaml'`.**

- Symptom: Config-loading import fails.
- Likely cause: `PyYAML` is missing from `requirements.txt`.
- Files to inspect: `requirements.txt`, config-loading functions.
- How to reproduce: Install only tracked requirements in a clean environment.
- How to fix: Install `PyYAML`; update `requirements.txt`.
- How to prevent recurrence: Add dependency completeness test.

**Scenario: `python src/Generator.py` fails with tuple/range error.**

- Symptom: Type error around `range(max_new_tokens)`.
- Likely cause: `max_new_tokens = 120,` creates tuple `(120,)`.
- Files to inspect: `src/Generator.py:69`.
- How to reproduce: Run `python src/Generator.py` after creating `model.pkl`.
- How to fix: Change to `max_new_tokens = 120`.
- How to prevent recurrence: Add script smoke test and main guard.

**Scenario: `python src/Generator.py` fails around `prompt_len`.**

- Symptom: Error that `prompt_len` cannot be accessed before assignment.
- Likely cause: Nested `generate()` reads `prompt_len` before outer function computes it.
- Files to inspect: `src/Generator.py:51-62`.
- How to reproduce: Run generation path.
- How to fix: Compute `prompt_len` before calling nested `generate()`, or pass it as an argument.
- How to prevent recurrence: Unit-test `generate_text()`.

**Scenario: Training raises "Not enough tokens".**

- Symptom: `ValueError("Not enough tokens to make even one (context_length+1) window.")`.
- Likely cause: Corpus has fewer than `context_length + 1` tokens after tokenization.
- Files to inspect: `src/Training_Loop.py:20-23`.
- How to reproduce: Use a tiny `train.txt` or very large context length.
- How to fix: Use more text, reduce context length, or adjust preprocessing.
- How to prevent recurrence: Validate token count before constructing model/training.

**Scenario: Pretrained loader shape copy fails.**

- Symptom: PyTorch `copy_` shape mismatch.
- Likely cause: Custom model dimensions do not match DistilGPT2.
- Files to inspect: `src/load_pretrained_model.py:6-12`, `src/load_pretrained_model.py:33-104`.
- How to reproduce: Change `model_dim`, `num_blocks`, or `num_heads`.
- How to fix: Restore DistilGPT2-compatible dimensions or update mapping for the chosen checkpoint.
- How to prevent recurrence: Add shape assertions and loader tests.

### 15.5 Design Tradeoff Questions And Answers

**Q: Simplicity vs scalability: what did the repo choose?**

A: It chooses simplicity and inspectability. Per-head attention modules and explicit loops are easier to understand but less efficient than fused QKV projections.

**Q: Local vs cloud assumptions?**

A: Training and generation assume local files and root-relative paths. README links a Streamlit deployment, but no cloud config is tracked. The project is more local-first than deploy-first.

**Q: Sync vs async behavior?**

A: Everything is synchronous: training loops, weight loading, and generation. This is simpler but not suited for concurrent serving.

**Q: Type safety tradeoff?**

A: The code uses `torchtyping.TensorType` annotations but does not enforce shapes at runtime. Some annotations are inaccurate, such as `batch_loader` returning tensors despite list-oriented annotation.

**Q: State management tradeoff?**

A: Streamlit session state is simple and enough for a demo. There is no database or persistent conversation storage, which avoids privacy/storage complexity but limits product behavior.

**Q: Error handling tradeoff?**

A: UI startup errors are user-friendly, but script errors mostly bubble up. This is acceptable for educational scripts but weak for production.

**Q: Testing choice tradeoff?**

A: The repo prioritizes implementation and documentation over tests. That speeds learning but leaves regressions easy to miss.

**Q: Framework/library choice tradeoff?**

A: PyTorch gives full control over model internals. Hugging Face is used only for loading reference DistilGPT2 weights. Streamlit gives a fast UI with minimal frontend code.

**Q: Performance choice tradeoff?**

A: The model is readable rather than optimized. It manually builds attention instead of using high-performance fused kernels.

### 15.6 Behavioral / STAR Stories

These stories are framed from repository evidence. Where the repo does not prove an event actually happened, the story is marked as suggested framing.

**Building the project**

- Situation: I wanted to demonstrate understanding of GPT-2 internals rather than only calling a pretrained model.
- Task: Build a decoder-only transformer in PyTorch with training and generation.
- Action: I split the architecture into embeddings, transformer blocks, multi-head attention, single-head causal attention, training loop, generation script, and Streamlit UI.
- Result: The repo shows the end-to-end path from text corpus to logits, loss, sampling, and a demo interface.

**Debugging a hard issue - suggested framing**

- Situation: A custom transformer and Hugging Face DistilGPT2 use different parameter layouts.
- Task: Make a pretrained checkpoint run inside my architecture.
- Action: I inspected the state dict layout, split fused QKV weights, sliced per head, transposed Conv1D weights into `nn.Linear` layout, copied all relevant parameters, and compared logits.
- Result: The loader validates the transferred model by printing max absolute logit difference and argmax IDs.

**Making an architectural decision**

- Situation: I needed the implementation to be understandable for learning/interviews.
- Task: Choose between concise fused attention and explicit per-head attention.
- Action: I implemented each head as its own module with separate query/key/value linears and a visible causal mask.
- Result: The code is easier to explain line by line, and the pretrained loader demonstrates how to map fused real-world weights into that educational layout.

**Improving reliability - suggested framing**

- Situation: Training transformer models can become unstable.
- Task: Add basic stabilization without adding a large training framework.
- Action: The training loop uses AdamW, optional CUDA AMP, and gradient clipping.
- Result: The code has practical training mechanics while staying compact.

**Learning a new tool/framework - suggested framing**

- Situation: I wanted a quick interactive demo.
- Task: Wrap the model in a web UI.
- Action: I used Streamlit, session state, sidebar controls, and chat input to create a text-continuation interface.
- Result: The project can be demoed interactively through `frontend/app.py`.

**Handling ambiguity**

- Situation: A chat UI could imply an instruction-tuned assistant, but the model is raw text continuation.
- Task: Present the demo honestly.
- Action: README and UI captions state that it is not a conversational bot/chat-tuned assistant.
- Result: The project avoids overclaiming and sets expectations correctly.

**Testing/validation**

- Situation: A from-scratch implementation needs evidence it matches real GPT behavior.
- Task: Validate architecture fidelity.
- Action: The loader compares custom logits against Hugging Face DistilGPT2 on the same tokenized prompt.
- Result: The repo has a sanity check, though not a complete automated test suite.

**Deployment/production readiness - suggested framing**

- Situation: The project has a Streamlit demo but is not production-ready.
- Task: Explain readiness honestly in an interview.
- Action: Point to the README limitations and identify concrete gaps: dependencies, tests, artifact management, CI, safer serialization.
- Result: The discussion shows engineering judgment, not just implementation enthusiasm.

### 15.7 "Explain This Project To..." Section

**A recruiter**

"It is a portfolio project where I built a GPT-2-style language model from scratch in PyTorch, including training, text generation, pretrained weight loading, and a web demo. It shows I understand machine learning systems below the API level."

**A non-technical user**

"This project is a small educational version of the technology behind text-generating AI. It learns from text and predicts what word or token should come next. There is also a simple web page where you can type a prompt and see the model continue it."

**A junior developer**

"Start with `src/GPT.py`: token IDs become embeddings, embeddings go through transformer blocks, and the model returns logits. Then read `Self_Attention_Mechanism.py` to see how each token looks back at previous tokens. Finally read `Training_Loop.py` to see how the model learns by predicting the next token."

**A senior engineer**

"This is a compact PyTorch transformer implementation with explicit architecture boundaries and a non-trivial checkpoint-conversion script. It is not productionized, but it demonstrates tensor-shape reasoning, attention masking, training mechanics, sampling, and parameter-layout translation from Hugging Face GPT-2."

**A product manager**

"The product surface is a demo that lets users type prompts and adjust generation settings. The value is educational and demonstrational, not a production assistant. The next product step would be making setup more reliable and clarifying what kind of outputs users should expect."

**A hiring manager**

"This project is useful interview evidence because it shows both implementation depth and honesty about limitations. The candidate built core ML logic manually, integrated a real pretrained model, and wrapped it in a demo, while documenting that it is not a scalable production LLM."

**An ML/AI engineer**

"The model is a decoder-only transformer with learned token/positional embeddings, pre-LN blocks, explicit per-head causal attention, GELU MLP, cross-entropy next-token training, and top-k/temperature sampling. The loader maps DistilGPT2's fused QKV Conv1D weights into separate per-head PyTorch linears and validates logits."

## 16. Glossary

| Term | Definition in this project |
|---|---|
| GPT | Generative Pretrained Transformer; here, a custom decoder-only transformer class in `src/GPT.py`. |
| Decoder-only transformer | Transformer architecture that uses causal self-attention and predicts next tokens autoregressively. |
| Token | Integer ID representing a piece of text from the GPT-2 tokenizer. |
| tiktoken | Tokenizer library used to encode/decode text with the GPT-2 vocabulary. |
| Context length | Maximum sequence length the model can process with its positional embedding table. |
| Embedding | Learned vector representation of token IDs or positions. |
| Positional embedding | Learned vector added to token embeddings to represent token position. |
| Logits | Raw unnormalized vocabulary scores returned by the model. |
| Softmax | Function converting logits into probabilities, used in attention and sampling. |
| Cross-entropy | Training loss used for next-token prediction. |
| Causal mask | Lower-triangular mask that blocks attention to future tokens. |
| Self-attention | Mechanism where tokens attend to other tokens in the same sequence. |
| Multi-head attention | Parallel attention heads whose outputs are concatenated and projected. |
| Q/K/V | Query, key, and value projections used in attention. |
| Head dimension | Per-head attention width, usually `model_dim / num_heads`. |
| Pre-layernorm | Transformer block design where layer norm is applied before attention and MLP sublayers. |
| Residual connection | Addition of sublayer output back to its input. |
| Feed-forward network | MLP inside transformer block, here `D -> 4D -> D` with GELU and dropout. |
| GELU | Activation function used in transformer MLP. |
| AdamW | Optimizer used by the training loop. |
| AMP | Automatic mixed precision; used on CUDA in training. |
| Gradient clipping | Limiting gradient norm for stability. |
| Top-k sampling | Sampling method that keeps only the highest-k logits before softmax. |
| Temperature | Scaling factor applied to logits before sampling. |
| EOS/EOT token | End-of-text token from tiktoken. |
| DistilGPT2 | Smaller GPT-2-like Hugging Face model used for pretrained weight transfer. |
| Conv1D layout | Hugging Face GPT-2 weight layout that differs from PyTorch `nn.Linear`. |
| Git LFS | Git Large File Storage, used for `train.txt` and configured for `*.pkl`. |
| `model.pkl` | Generated scratch-trained pickled model expected by `src/Generator.py`. |
| `gpt_weights.pt` | Generated scratch-trained state dict. |
| `pre_trained_model.pkl` | Generated pickled custom model after DistilGPT2 weight transfer, expected by Streamlit. |
| Streamlit | Python UI framework used by `frontend/app.py`. |
| `st.session_state.messages` | Streamlit session-local chat history list. |

## 17. Risks, Gaps, And Improvement Roadmap

### Highest-risk code areas

| Risk area | Evidence | Why it matters |
|---|---|---|
| Pickle loading | `src/Generator.py`, `frontend/app.py` | Unsafe for untrusted artifacts and brittle across code changes. |
| Pretrained weight mapping | `src/load_pretrained_model.py` | Shape/layout-sensitive and not covered by automated tests. |
| Standalone generator | `src/Generator.py` | Contains apparent runtime bugs. |
| Dependency manifest | `requirements.txt` | Missing imported packages and no version pins. |
| Path assumptions | Multiple files open root-relative paths. | Commands fail from unexpected working directories. |
| No tests | Whole repo | Regressions likely in tensor logic. |

### Missing tests

- Causal mask correctness.
- Shape consistency across model modules.
- Token-window target shifting.
- Training smoke test.
- Generation sampling behavior.
- Streamlit prompt builder.
- DistilGPT2 mapping on synthetic weights.
- Import/dependency smoke test.

### Security concerns

- Pickle loading.
- No model artifact integrity checks.
- No authentication if deployed publicly through Streamlit.
- No rate limiting or abuse protection.
- No explicit privacy handling for user prompts.

### Performance concerns

- Per-head Python loop.
- Mask allocation each forward pass.
- No KV cache for generation, so each new token recomputes full context.
- Full token tensor can be moved to GPU for training.
- No batching/caching in Streamlit generation.

### Maintainability concerns

- Duplicate config loader code.
- Hard-coded model/training/generation settings.
- Direct imports instead of package structure.
- Generated artifacts not fully documented as part of setup/deploy.
- No CI.

### Documentation gaps

- Python version.
- Exact dependency versions.
- Corpus provenance/license.
- Expected generated artifact sizes.
- Expected validation output/tolerance for DistilGPT2 comparison.
- Known bugs in `src/Generator.py`.
- Deployment artifact handling.

### Suggested improvements ordered by impact

1. Add complete, pinned dependencies including `streamlit` and `PyYAML`.
2. Add tests for attention masking, shape flow, preprocessing shifts, and generation.
3. Fix `src/Generator.py` runtime issues and add main guard.
4. Replace pickle loading with state dict plus explicit config where feasible.
5. Centralize configuration and path resolution.
6. Add a tiny smoke-test workflow for CI.
7. Add expected output/tolerance documentation for pretrained validation.
8. Add optional `st.cache_resource` for model loading in Streamlit.
9. Add corpus provenance and data documentation.
10. Add performance improvements such as cached masks or vectorized attention if moving beyond educational scope.

### Suggested improvements ordered by effort

1. Update `requirements.txt` with `streamlit` and `PyYAML`.
2. Fix `max_new_tokens = 120,` in `src/Generator.py`.
3. Move `prompt_len` before nested generation call.
4. Add `if __name__ == "__main__"` guard to `src/Generator.py` and `src/load_pretrained_model.py`.
5. Add assertions for `model_dim % num_heads`.
6. Add README note about running commands from repo root.
7. Add small unit tests for `build_prompt()` and token shifting.
8. Centralize `load_hyperparameters`.
9. Add path constants based on project root.
10. Replace pickle artifacts with state dict loading.

## 18. Coverage Checklist

### Inventory summary

- Total tracked files analyzed: 18.
- Total tracked folders analyzed: 3 including repository root, `frontend/`, and `src/`.
- Non-root tracked folders analyzed: 2 (`frontend/`, `src/`).
- Notable untracked files observed: none reported by `git ls-files --others --exclude-standard`.
- Git status limitation: `git status --short` produced Git LFS clean-filter warnings involving `.git/lfs/tmp` and reported `M train.txt`; this document treats `train.txt` as a tracked LFS corpus and does not infer application behavior from that status warning.
- Source-code files with meaningful deep dives: 9 Python files.
- Binary/large files covered at high level: `image.png`, `train.txt`.
- Files skipped: none among tracked files.

### Files covered in the deep dive

| Covered | Path | Coverage level | Reason |
|---|---|---|---|
| Yes | `.gitattributes` | Config deep dive | Tracked Git LFS config. |
| Yes | `.gitignore` | Config deep dive | Tracked ignore config. |
| Yes | `LICENSE` | Legal/documentation deep dive | Tracked license file. |
| Yes | `README.md` | Documentation deep dive | Primary project docs. |
| Yes | `config.yaml` | Config deep dive | Runtime tokenizer setting. |
| Yes | `frontend/app.py` | Source deep dive | Streamlit app and generation UI. |
| Yes | `image.png` | High-level binary artifact | Binary screenshot; code-level analysis not applicable. |
| Yes | `requirements.txt` | Config deep dive | Dependency manifest. |
| Yes | `src/Data_Preprocessing.py` | Source deep dive | Tokenization and batch helpers. |
| Yes | `src/GPT.py` | Source deep dive | Top-level model. |
| Yes | `src/Generator.py` | Source deep dive | Standalone generation script. |
| Yes | `src/Multi_Headed_Attention.py` | Source deep dive | Multi-head attention module. |
| Yes | `src/Self_Attention_Mechanism.py` | Source deep dive | Causal attention head. |
| Yes | `src/Training_Loop.py` | Source deep dive | Training loop. |
| Yes | `src/Transformer_Block.py` | Source deep dive | Transformer block and MLP. |
| Yes | `src/load_pretrained_model.py` | Source deep dive | DistilGPT2 weight transfer. |
| Yes | `src/run_train.py` | Source deep dive | Scratch training entrypoint. |
| Yes | `train.txt` | High-level large artifact | Large tracked LFS corpus; line-by-line analysis not applicable. |

### Files only covered at high level

| Path | Reason |
|---|---|
| `image.png` | Binary PNG screenshot. Documented role, size, dimensions, and usage; detailed code-level analysis is not applicable. |
| `train.txt` | Large text corpus tracked with Git LFS. Documented role, size, line count, apparent content type, and training usage; exhaustive content analysis would not help explain code behavior. |

### Files skipped

None.

### Validation checklist

- Markdown file exists at repository root: this file, `GPT-built-from-scratch_ALLINFO.md`.
- Filename follows `<project_name>_ALLINFO.md`: project name inferred as `GPT-built-from-scratch`.
- Every tracked file appears in the repository map and coverage checklist.
- Every source-code file has a meaningful deep-dive entry.
- Secret values were not copied; no tracked secret values were observed.
- Quick-start/run/test/deploy instructions are based on README, source imports, and script entrypoints.
- Interview questions are specific to this repository's files, functions, and architecture.
- This file is broader than a README rewrite: it includes workflows, code chunk walkthroughs, risks, test gaps, operations, extension guidance, and interview preparation.
