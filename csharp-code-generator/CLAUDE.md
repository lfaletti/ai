# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An **educational** proof-of-concept that builds a C#-code generator from scratch to teach how Copilot-style tools work: a specialized C# tokenizer, a 4-layer decoder-only Transformer (with Fill-in-the-Middle), and a FAISS-based RAG retrieval layer — plus a static HTML dashboard that visualizes all of it. The code is written for teaching, with **extensive Spanish-language comments and Spanish/emoji print output**. When editing, preserve that didactic style — clarity for a learner outranks brevity. `README.md` is a long-form tutorial covering the *why* behind every design choice.

This is one project inside a multi-project repo (`csharp-code-generator/`, `mini-transformer/`, `chat-agent/`, etc.); each project is self-contained with its own deps.

## Commands

```bash
pip install -r requirements.txt   # torch, numpy, faiss-cpu

python main.py                    # Full pipeline: dataset → tokenizer → train (5 epochs) →
                                  #   RAG index → usage examples → export visualization JSON
python main.py --rapido           # Same pipeline, 2 epochs (quick smoke run)
python main.py --epochs 10        # Control epoch count

# View the dashboard AFTER running main.py (it needs the exported datos_*.json):
python -m http.server 8080 -d visualizacion   # then open http://localhost:8080
```

There is **no test suite, linter, or build step** — a `--rapido` run is the closest thing to a smoke test. Note: the `main.py` module docstring mentions a `--solo-viz` flag that is **not actually implemented**; only `--rapido` and `--epochs` exist.

## Architecture

`main.py` is the orchestrator — it runs the system as six sequential `paso_N_*` functions (dataset → tokenizer → train → RAG → examples → export). Reading it top-to-bottom is the fastest way to see how the pieces connect. The four subsystems:

- **`datos/`** — `dataset_csharp.py` generates ~100 synthetic C# snippets (Controllers, Services, DTOs, LINQ, async); `base_conocimiento.py` builds the separate knowledge base that RAG indexes. Note these are **two distinct corpora**: the model trains on the dataset, RAG retrieves from the knowledge base.
- **`modelo/`** — `tokenizador_csharp.py` (`TokenizadorCSharp`, code-aware: `=>`/`List<` are atomic tokens, PascalCase is split, explicit `<INDENT>`/`<DEDENT>`), `transformer.py` (`TransformerCodeGen` + `ConfiguracionTransformer`), `entrenamiento.py` (`EntrenadorModelo` + `ConfigEntrenamiento` — owns training **and** the inference API).
- **`rag/`** — `embeddings.py` (TF-IDF + hashing → 256-dim vectors, **not** a learned embedding) and `sistema_rag.py` (`SistemaRAG`: FAISS index + retrieval).
- **`utils/exportar_visualizacion.py`** — `ExportadorVisualizacion` writes the `datos_*.json` files the dashboard reads.

### Key design points (non-obvious)

- **Fill-in-the-Middle is central, not optional.** 30% of training examples (`proporcion_fim=0.3`) are reformatted with `<FIM_PREFIX>`/`<FIM_SUFFIX>`/`<FIM_MIDDLE>` special tokens so the model learns to complete *inside* a file given both sides. The tokenizer must define these special tokens for FIM to work — they are coupled.
- **Inference entry points** live on `EntrenadorModelo`: `completar_codigo(prompt, max_tokens, temperatura)` for left-to-right completion and `completar_fim(prefijo, sufijo, ...)` for infill. RAG comparison runs through `SistemaRAG.comparar_con_sin_rag(query, modelo, tokenizador, ...)`.
- **RAG embeddings are deterministic TF-IDF, not neural.** Retrieval quality depends on lexical overlap with the knowledge base — keep that in mind when results look off; it's expected behavior for this PoC, not a bug.
- **The model and the visualizer are coupled through the JSON shape.** `ExportadorVisualizacion` emits `visualizacion/datos_*.json` (tokenizacion, atencion, generacion, rag, comparacion, entrenamiento) and `visualizacion/index.html` reads them. **If you change what gets exported, update `index.html` to match.** The `datos_*.json` and `index.html` are committed as the working demo (treated as source, not generated artifacts).
- **Attention capture is opt-in:** the model only records attention weights when `ConfiguracionTransformer(guardar_atencion=True)` — the export step relies on this being set in `main.py`.

### Generated vs. committed

`main.py` recreates `checkpoints/` (`vocabulario.json` + `modelo_epoch_*.pt`, ~42 MB each). These are **gitignored** (see root `.gitignore`: `checkpoints/`, `*.pt`) — the whole pipeline is reproducible from source, so don't commit weights.

## Conventions

- Default model config: `d_modelo=256, num_cabezas=8, num_capas=4, dim_ff=1024, max_longitud=512, dropout=0.1`, vocab cap 4000. Training: `learning_rate=3e-4, batch_size=4, warmup_steps=50`, sequence length truncated to 256.
- Reproducibility: `main.py` calls `fijar_semillas(42)` seeding random/numpy/torch — keep runs deterministic unless there's a reason not to.
- Identifiers, comments, and print statements are in Spanish with emoji prefixes — match this when adding code or output.
