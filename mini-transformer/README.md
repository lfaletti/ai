# 🐉 Mini Transformer Educativo

Un **Transformer decoder-only completo** implementado desde cero en PyTorch, con comentarios extensos en español que explican cada componente, por qué es necesario y cómo fluyen los datos.

## 📁 Estructura del Proyecto

```
mini_transformer/
├── mini_transformer.py          # 🧠 Implementación completa del modelo
├── train.py                     # 🚀 Script de entrenamiento + generación
├── visualize_transformer.html   # 📊 Visualización interactiva
├── README.md                    # 📖 Este archivo
├── modelo_entrenado.pt          # 💾 (generado) Modelo entrenado
└── trace_data.json              # 📈 (generado) Datos para visualización
```

## 🚀 Inicio Rápido

### 1. Requisitos

```bash
pip install torch
```

Solo necesitas PyTorch. No hay otras dependencias externas.

### 2. Entrenar el modelo

```bash
cd mini-transformer
python train.py
```

Esto:
- Entrena el modelo en un cuento corto en español (~50 épocas)
- Muestra el progreso con loss y perplexity
- Genera muestras de texto durante el entrenamiento
- Guarda el modelo en `modelo_entrenado.pt`
- Exporta datos de traza para visualización en `trace_data.json`

### 3. Demo interactiva (CLI)

```bash
python train.py demo
```

Escribe un prompt y el modelo lo continuará.

### 4. Visualización interactiva

Abre `visualize_transformer.html` en tu navegador y carga el archivo `trace_data.json` generado durante el entrenamiento. La visualización incluye:

- **🏗️ Arquitectura**: Diagrama SVG del modelo completo
- **⚡ Flujo de Datos**: Paso a paso por cada componente
- **👁️ Attention Weights**: Mapas de calor de la atención
- **✨ Generación**: Animación token por token
- **📈 Entrenamiento**: Curvas de loss y perplexity

> La visualización funciona también sin cargar datos (usa datos de demostración integrados).

## 🧠 Arquitectura del Modelo

```
Tokens → Embedding → + Positional Encoding → [Decoder Layer × N] → LayerNorm → Linear → Softmax
                                                    │
                                          ┌─────────┴──────────┐
                                          │  LayerNorm          │
                                          │  Multi-Head Attn    │
                                          │  + Residual         │
                                          │  LayerNorm          │
                                          │  Feed-Forward       │
                                          │  + Residual         │
                                          └────────────────────┘
```

### Componentes implementados:

| Componente | Archivo | Descripción |
|---|---|---|
| `PositionalEncoding` | `mini_transformer.py` | Codificación posicional con sin/cos |
| `scaled_dot_product_attention` | `mini_transformer.py` | Atención escalada Q·K^T/√d_k · V |
| `MultiHeadAttention` | `mini_transformer.py` | Múltiples cabezas de atención en paralelo |
| `FeedForward` | `mini_transformer.py` | Red FFN con GELU y expansión 4× |
| `DecoderLayer` | `mini_transformer.py` | Capa completa con Pre-LN y residuals |
| `MiniTransformer` | `mini_transformer.py` | Modelo completo con weight tying |
| `CharTokenizer` | `mini_transformer.py` | Tokenizer a nivel de carácter |
| `TextDataset` | `mini_transformer.py` | Dataset con ventanas deslizantes |

### Hiperparámetros por defecto:

| Parámetro | Valor | Descripción |
|---|---|---|
| `d_model` | 128 | Dimensión de embeddings |
| `num_heads` | 4 | Cabezas de atención |
| `num_layers` | 4 | Capas del decoder |
| `d_ff` | 512 | Dimensión FFN (4 × d_model) |
| `seq_len` | 64 | Longitud de secuencia |
| `dropout` | 0.1 | Regularización |

## 📚 Conceptos Clave Explicados en el Código

### Positional Encoding
> Sin él, "el gato persigue al ratón" = "el ratón persigue al gato". Usa funciones sin/cos de diferentes frecuencias para crear un fingerprint único por posición.

### Self-Attention (Q, K, V)
> **Query** = "¿qué busco?", **Key** = "¿qué información tengo?", **Value** = "mi contenido real". El producto punto Q·K mide compatibilidad, softmax normaliza, y se pondera V.

### Multi-Head Attention
> Múltiples "expertos" que atienden a diferentes tipos de relaciones (sintácticas, semánticas, posicionales...) simultáneamente.

### Feed-Forward Network
> Añade no-linealidad. La atención solo mezcla información linealmente; el FFN permite transformaciones complejas.

### Conexiones Residuales
> `output = x + SubLayer(x)`. Permiten al gradiente fluir directamente, evitando el problema del gradiente que se desvanece.

### Layer Normalization
> Normaliza valores a media=0, varianza=1 dentro de cada ejemplo. Estabiliza el entrenamiento.

### Máscara Causal
> Matriz triangular inferior que impide que el token i vea tokens j > i. Esencial para generación autoregresiva.

### Weight Tying
> Compartir pesos entre embedding de entrada y capa de salida. Reduce parámetros y mejora rendimiento.

## 🔧 API del Modelo

```python
from mini_transformer import MiniTransformer, CharTokenizer

# Crear modelo
model = MiniTransformer(vocab_size=50, d_model=128, num_heads=4, num_layers=4)

# Guardar / Cargar
model.save("mi_modelo.pt")
model = MiniTransformer.load("mi_modelo.pt")

# Generar texto
prompt_ids = torch.tensor([[1, 2, 3, 4]])
generated = model.generate(prompt_ids, max_new_tokens=100, temperature=0.7, top_p=0.9)

# Generar con traza (para visualización)
trace = model.generate_with_trace(prompt_ids, max_new_tokens=10)
```

## 📊 Parámetros de Generación

| Parámetro | Rango | Efecto |
|---|---|---|
| `temperature` | 0.1 - 2.0 | Baja = conservador, Alta = creativo |
| `top_k` | 1 - vocab_size | Solo considerar los k más probables |
| `top_p` | 0.0 - 1.0 | Nucleus sampling (p=0.9 = top 90% de masa) |

## 🔒 Reproducibilidad

**⚠️ Por defecto, `train.py` NO fija semillas.** Esto significa que cada ejecución producirá un modelo diferente debido a múltiples fuentes de aleatoriedad:

| Fuente de aleatoriedad | Dónde ocurre | Cómo fijarla |
|---|---|---|
| Inicialización de pesos | `nn.init.xavier_uniform_` | `torch.manual_seed(seed)` |
| Orden de datos | `DataLoader(shuffle=True)` | `DataLoader(generator=g)` con `g.manual_seed(seed)` |
| Dropout | `nn.Dropout(p=0.1)` en atención, FFN, PE | `torch.manual_seed(seed)` |
| Sampling de tokens | `torch.multinomial` en `generate()` | `torch.manual_seed(seed)` |
| NumPy / Python random | Uso interno | `np.random.seed(seed)` / `random.seed(seed)` |
| cuDNN (GPU) | Algoritmos no deterministas | `torch.backends.cudnn.deterministic = True` |

### Demo interactiva

Ejecuta la demo para ver la diferencia en vivo:

```bash
python demo_reproducibilidad.py
```

Esta demo:
1. Entrena 2 veces **sin semilla** → muestra que los pesos finales son **diferentes**
2. Entrena 2 veces **con semilla fija** → muestra que son **idénticos**
3. Genera texto con y sin semilla para demostrar que el sampling también importa
4. Entrena con dos semillas distintas → muestra que cada semilla da un camino diferente

### Código mínimo para reproducibilidad

```python
import torch, random, numpy as np

def fijar_semillas(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Llamar ANTES de crear el modelo y el DataLoader
fijar_semillas(42)

# Para el DataLoader:
g = torch.Generator()
g.manual_seed(42)
DataLoader(dataset, shuffle=True, generator=g)
```

> **Nota:** La reproducibilidad exacta solo se garantiza dentro del mismo entorno (misma versión de PyTorch, mismo hardware CPU/GPU, mismo sistema operativo). Entre CPU y GPU, o entre versiones distintas de CUDA, los resultados pueden variar ligeramente.

## 🎓 Para Aprender Más

1. **Paper original**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
2. **Blog ilustrado**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar)
3. **Nanoprion de Karpathy**: [nanoGPT](https://github.com/karpathy/nanoGPT) — implementación minimalista de GPT
4. **Video**: [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Andrej Karpathy)
