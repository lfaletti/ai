# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An **educational** decoder-only Transformer (GPT-style) implemented from scratch in PyTorch. The code is written for teaching: extensive Spanish-language comments explain *what* each component does, *why* it's needed, and *how* tensor dimensions flow. When editing, **preserve the didactic comment style and Spanish prose** — clarity for a learner is the priority over brevity or cleverness.

The only runtime dependency is `torch` (`pip install torch`). No requirements file exists.

## Commands

```bash
python train.py                  # Train on the built-in Spanish story (~50 epochs),
                                 # writes modelo_entrenado.pt and trace_data.json
python train.py demo             # Interactive REPL: type a prompt, model continues it
                                 # (requires modelo_entrenado.pt to exist)

python demo_reproducibilidad.py  # Shows how seeds make training deterministic (trains its own models)
python demo_sampling_seed.py     # Isolates the single random line in generation (torch.multinomial)
python comparar_sampling.py      # Same model+prompt, different seeds → different text

python mini_transformer.py       # Sanity check: builds a model and runs one forward pass
```

There is **no test suite, linter, or build step**. The `__main__` block of `mini_transformer.py` is the closest thing to a smoke test. `modelo_entrenado.pt` and `trace_data.json` are generated artifacts (committed but reproducible via `train.py`).

## Architecture

Everything model-related lives in `mini_transformer.py`, composed bottom-up:

`PositionalEncoding` → `scaled_dot_product_attention` (free function) → `MultiHeadAttention` → `FeedForward` → `DecoderLayer` → `MiniTransformer`. Plus `CharTokenizer` (character-level) and `TextDataset` (sliding windows, target = input shifted +1).

Key design decisions baked into the code:
- **Pre-LayerNorm** (norm before each sublayer, GPT-2 style), not Post-LN.
- **Weight tying**: `output_projection.weight` is shared with `token_embedding.weight` when dimensions match — editing one affects both.
- **Causal masking** is generated inside `forward` via `_generate_causal_mask`; attention applies it as `masked_fill(mask == 0, -inf)`.
- A model checkpoint (`.pt`) bundles `config` + `state_dict` + the tokenizer vocab (`token_to_id`/`id_to_token`), so `MiniTransformer.load()` fully reconstructs a usable model without the original tokenizer object.

### Two generation paths
- `generate()` — fast autoregressive sampling with `temperature` / `top_k` / `top_p` (nucleus). `@torch.no_grad()`.
- `generate_with_trace()` — same loop but records per-step top-5 predictions and per-layer attention weights into a dict. This dict (plus training history and tokenizer) is serialized to `trace_data.json`, which `visualize_transformer.html` loads in the browser. **If you change the trace dict shape, update the HTML visualizer to match.**

### The single source of randomness
Text generation is fully deterministic except for `torch.multinomial(...)` in `generate()` (~line 902). The three demo scripts exist specifically to teach this — they pivot on `torch.manual_seed()` controlling that one call. Training has additional randomness sources (weight init, `DataLoader(shuffle=True)`, dropout); see the reproducibility table in `README.md`. **`train.py` intentionally does NOT set seeds** — don't add seeding to it without reason, as the reproducibility demo depends on it being non-deterministic.

## Conventions

- Default hyperparameters: `d_model=128, num_heads=4, num_layers=4, d_ff=512` (d_ff = 4×d_model), `seq_len=64`, `dropout=0.1`. `d_model` must be divisible by `num_heads` (asserted).
- User-facing print statements use emoji prefixes and Spanish — match this when adding output.
