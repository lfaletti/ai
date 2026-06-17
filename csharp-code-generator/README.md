# 🔧 Generador de Código C# — Prueba de Concepto Educativa

> **Un Transformer de 4 capas + RAG con FAISS para entender cómo funcionan los generadores de código estilo Copilot**

---

## 📋 Índice

1. [¿Qué es este proyecto?](#-qué-es-este-proyecto)
2. [Diferencias técnicas vs modelos de texto natural](#-diferencias-técnicas-vs-modelos-de-texto-natural)
3. [Arquitectura completa paso a paso](#-arquitectura-completa-paso-a-paso)
4. [Cómo funciona RAG en generadores de código](#-cómo-funciona-rag-en-generadores-de-código)
5. [Cuándo RAG mejora la generación](#-cuándo-rag-mejora-la-generación)
6. [Estructura del proyecto](#-estructura-del-proyecto)
7. [Instalación y uso](#-instalación-y-uso)
8. [Ejemplos de uso](#-ejemplos-de-uso)
9. [Visualización interactiva](#-visualización-interactiva)

---

## 🎯 ¿Qué es este proyecto?

Este proyecto es una **prueba de concepto educativa** que implementa desde cero un generador de código C# para entender los mecanismos internos de herramientas como GitHub Copilot, Amazon CodeWhisperer, o Tabnine.

### Componentes principales:

| Componente | Descripción | Archivo(s) |
|---|---|---|
| **Tokenizador C#** | Tokenización especializada con PascalCase, operadores lambda, genéricos | `modelo/tokenizador_csharp.py` |
| **Transformer 4 capas** | Modelo decoder-only con soporte FIM | `modelo/transformer.py` |
| **Dataset sintético** | 100+ snippets de C# profesional (Controllers, Services, DTOs, LINQ) | `datos/dataset_csharp.py` |
| **Sistema RAG** | Búsqueda vectorial con FAISS + base de conocimiento de código | `rag/sistema_rag.py` |
| **Visualización HTML/JS** | Dashboard interactivo con attention maps, tokenización, flujo RAG | `visualizacion/index.html` |

---

## 🔬 Diferencias Técnicas vs Modelos de Texto Natural

Esta es la pregunta central del proyecto: **¿por qué un generador de código NO es simplemente un modelo de texto entrenado con código?**

### 1. Tokenización Especializada

| Aspecto | Texto Natural (GPT) | Código C# (nuestro modelo) |
|---|---|---|
| **Operadores** | `=>` se rompe en `=` + `>` | `=>` es un token único (operador lambda) |
| **PascalCase** | `GetAllProducts` = 1 token raro | `Get` + `All` + `Products` = 3 tokens frecuentes |
| **Genéricos** | `List<string>` = tokens aleatorios | `List<` es un token de tipo genérico |
| **Indentación** | Ignorada | Tokens explícitos `<INDENT>` / `<DEDENT>` |
| **Keywords** | Tratadas como palabras normales | IDs fijos con semántica especial |

**¿Por qué importa?** Si el tokenizador rompe `=>` en dos tokens, el modelo tiene que APRENDER que `=` seguido de `>` significa "lambda". Con un tokenizador especializado, `=>` ya es atómico — el modelo no necesita aprender esta combinación.

### 2. Vocabulario

```
Texto natural:  ~50,000 tokens (palabras comunes + subpalabras de idiomas)
Código C#:      ~4,000-8,000 tokens (keywords + operadores + identificadores frecuentes)
```

La distribución es MUY diferente:
- En texto, "the" es el token más frecuente
- En código C#, `{` y `}` son los tokens más frecuentes
- Keywords como `public`, `class`, `return` dominan la cabeza de la distribución
- Los nombres de variables son la "cola larga" (aparecen pocas veces)

### 3. Estructura Jerárquica

El código C# tiene una **jerarquía estricta**:

```
namespace → class → method → statement → expression
```

El modelo debe aprender que `}` en la línea 50 cierra la `{` de la línea 10. Esta **dependencia a larga distancia** es mucho más estricta que en texto natural, donde cerrar un párrafo es flexible.

### 4. Fill-in-the-Middle (FIM)

Los modelos de texto generan de izquierda a derecha. Los generadores de código necesitan completar **en medio de un archivo**:

```csharp
public int GetTotal(List<int> items)
{
    // ← CURSOR AQUÍ: el modelo debe generar el body
    //   viendo TANTO el código anterior COMO el return de abajo
    return total;
}
```

**Formato FIM:**
```
<FIM_PREFIX>public int GetTotal(...) { <FIM_SUFFIX>return total; } <FIM_MIDDLE>var total = items.Sum();
```

El modelo ve PREFIX + SUFFIX y genera MIDDLE. Esto NO existe en modelos de texto estándar.

### 5. Coherencia de Tipos

En texto natural, "El gato está en el ___" acepta muchas palabras. En C#:

```csharp
int total = ___  // SOLO puede ser una expresión que retorne int
```

El modelo debe aprender la **coherencia de tipos**: si declaro `int`, el valor asignado debe ser compatible con `int`. Esto requiere tracking de contexto más preciso que en texto natural.

---

## 🏗️ Arquitectura Completa Paso a Paso

### Flujo de Entrenamiento

```
1. Dataset Sintético         2. Tokenización C#          3. Entrenamiento Transformer
┌──────────────────┐       ┌──────────────────┐        ┌──────────────────────┐
│ Controllers      │       │ "public class" →  │        │ Embedding (256 dim)  │
│ Services         │──────▶│ [45, 23, 102]     │───────▶│ Attention × 4 capas  │
│ DTOs, LINQ       │       │ + ejemplos FIM    │        │ FFN + Residual       │
│ async/await      │       │ (30% del dataset) │        │ Loss: CrossEntropy   │
└──────────────────┘       └──────────────────┘        └──────────────────────┘
```

### Flujo de Inferencia (con RAG)

```
┌─────────────────────────────────────────────────────────────────┐
│  Desarrollador escribe: "// crear endpoint REST para pedidos"   │
│                                                                  │
│  ┌─── RAG Pipeline ───────────────────────────────────────────┐ │
│  │ 1. Query → Embedding (TF-IDF + hashing, 256 dim)          │ │
│  │ 2. FAISS busca top-3 snippets más similares                │ │
│  │ 3. Recupera: OrdersController, OrderService, OrderDto      │ │
│  │ 4. Combina como contexto antes del prompt                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── Transformer ────────────────────────────────────────────┐ │
│  │ Input: [contexto_RAG] + [prompt_usuario]                   │ │
│  │ Attention ve los ejemplos recuperados                       │ │
│  │ Genera token por token con temperature=0.7, top-k=50       │ │
│  │ Output: Código C# coherente con el proyecto                │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Detalle del Transformer (4 capas)

```
                 Input: [public, class, Product, {, ...]
                              │
                 ┌────────────▼────────────┐
                 │  Embedding de Tokens     │  Cada token → vector de 256 dims
                 │  + Embedding Posicional  │  Posición en la secuencia
                 └────────────┬────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │  Capa 1: Multi-Head Attention (8 cabezas)   │
          │  + Feed-Forward (256→1024→256) + Residual   │
          │  Aprende: relaciones locales (tipo↔variable) │
          ├───────────────────┬───────────────────┤
          │  Capa 2: Aprende: bloques {  }               │
          ├───────────────────┬───────────────────┤
          │  Capa 3: Aprende: relaciones método/clase    │
          ├───────────────────┬───────────────────┤
          │  Capa 4: Aprende: patrones arquitectónicos   │
          └───────────────────┬───────────────────┘
                              │
                 ┌────────────▼────────────┐
                 │  Head de Lenguaje       │  256 → 4000 (vocabulario)
                 │  softmax → probabilidad │  de cada posible token
                 └────────────┬────────────┘
                              │
                 Output: token más probable → "Product"
```

---

## 🔍 Cómo Funciona RAG en Generadores de Código

### ¿Qué es RAG?

**RAG = Retrieval-Augmented Generation** (Generación Aumentada por Recuperación)

Es una técnica que **mejora los modelos generativos** dándoles acceso a una base de conocimiento externa. En lugar de generar "de memoria", el modelo CONSULTA código existente antes de generar código nuevo.

### Analogía

| Sin RAG | Con RAG |
|---|---|
| Programador junior que escribe de memoria | Programador senior que revisa el código existente del proyecto antes de escribir |
| Puede inventar convenciones incorrectas | Mantiene consistencia con el proyecto |
| No conoce las APIs del proyecto | Usa las mismas APIs y patrones |

### Pipeline RAG Paso a Paso

```
1. INDEXACIÓN (offline, una vez)
   ┌─────────────────────────────────────────────┐
   │  Para cada snippet de código en el proyecto: │
   │  código → preprocesar → TF-IDF → embedding  │
   │  embedding → agregar al índice FAISS         │
   └─────────────────────────────────────────────┘

2. BÚSQUEDA (online, por cada petición)
   ┌─────────────────────────────────────────────┐
   │  Query: "crear endpoint REST para productos" │
   │  query → preprocesar → TF-IDF → embedding   │
   │  embedding → buscar en FAISS → top-3 vecinos │
   │  Resultado: ProductsController (sim: 0.85),  │
   │            OrdersController (sim: 0.72), ...  │
   └─────────────────────────────────────────────┘

3. GENERACIÓN (online, después de la búsqueda)
   ┌─────────────────────────────────────────────┐
   │  Contexto = snippets recuperados             │
   │  Input = [Contexto] + [Prompt del usuario]   │
   │  Transformer genera código informado         │
   │  por los ejemplos del proyecto               │
   └─────────────────────────────────────────────┘
```

### ¿Por qué FAISS?

FAISS (Facebook AI Similarity Search) permite búsqueda eficiente de vectores similares:

- **Sin FAISS**: comparar con 1M de snippets = 1M operaciones → lento
- **Con FAISS**: usa índices optimizados → ~O(log N) → rápido
- Para nuestro prototipo (~100 snippets), la diferencia es mínima, pero en producción es **esencial**

---

## ✅ Cuándo RAG Mejora la Generación

### RAG es MUY útil cuando:

1. **El proyecto tiene convenciones específicas**
   - Todos los Controllers heredan de `BaseController`
   - Las respuestas siempre usan `ApiResponse<T>`
   - RAG recupera estos patrones y el modelo los sigue

2. **Se necesita coherencia con código existente**
   - Si el proyecto usa `IOrderService`, el nuevo código NO debería inventar `OrderManager`
   - RAG asegura que se usen los mismos nombres e interfaces

3. **Se usan APIs o librerías específicas**
   - Si el proyecto usa FluentValidation, RAG recupera validadores existentes
   - El modelo genera validadores coherentes con el patrón del proyecto

4. **Se quiere reducir "alucinaciones"**
   - Sin RAG, el modelo puede inventar métodos que no existen
   - Con RAG, el modelo ve los métodos reales y los usa

### RAG es MENOS útil cuando:

1. **El código es genérico** (Hello World, algoritmos estándar)
2. **No hay código existente** como referencia (proyecto nuevo)
3. **El contexto del modelo es suficiente** (completar una línea simple)

---

## 📂 Estructura del Proyecto

```
csharp_code_generator/
├── main.py                          # Pipeline principal
├── requirements.txt                 # Dependencias Python
├── README.md                        # Este archivo
│
├── datos/                           # Módulo de datos
│   ├── __init__.py
│   ├── dataset_csharp.py            # Generador de dataset sintético C#
│   └── base_conocimiento.py         # Base de conocimiento para RAG
│
├── modelo/                          # Módulo del modelo
│   ├── __init__.py
│   ├── tokenizador_csharp.py        # Tokenizador especializado para C#
│   ├── transformer.py               # Modelo Transformer de 4 capas
│   └── entrenamiento.py             # Pipeline de entrenamiento e inferencia
│
├── rag/                             # Módulo RAG
│   ├── __init__.py
│   ├── embeddings.py                # Generador de embeddings (TF-IDF)
│   └── sistema_rag.py               # Sistema RAG completo con FAISS
│
├── utils/                           # Utilidades
│   ├── __init__.py
│   └── exportar_visualizacion.py    # Exportador de datos para HTML/JS
│
├── visualizacion/                   # Dashboard interactivo
│   ├── index.html                   # Visualización HTML/JS completa
│   └── datos_*.json                 # Datos generados por main.py
│
├── examples/                        # Ejemplos de uso
│   └── ejemplos_uso.py              # Ejemplos prácticos documentados
│
└── checkpoints/                     # Modelos entrenados (generado)
    ├── vocabulario.json
    └── modelo_epoch_*.pt
```

---

## 🚀 Instalación y Uso

### Requisitos

- Python 3.9+
- PyTorch 2.0+
- FAISS (faiss-cpu)

### Instalación

```bash
cd csharp_code_generator
pip install -r requirements.txt
```

### Ejecución

```bash
# Pipeline completo (5 epochs)
python main.py

# Entrenamiento rápido (2 epochs) - para probar
python main.py --rapido

# Controlar epochs
python main.py --epochs 10
```

### Ver la visualización

```bash
# Después de ejecutar main.py:
cd visualizacion
python -m http.server 8080
# Abrir http://localhost:8080 en el navegador
```

---

## 💡 Ejemplos de Uso

### 1. Autocompletado de clase

```python
resultado = entrenador.completar_codigo(
    "public class ProductService : IProductService\n{",
    max_tokens=100,
    temperatura=0.7,
)
print(resultado["codigo_generado"])
```

### 2. Generación desde comentario XML

```python
resultado = entrenador.completar_codigo(
    "/// <summary>\n/// Obtiene un producto por su ID.\n/// </summary>\npublic async",
    max_tokens=80,
)
```

### 3. Fill-in-the-Middle

```python
resultado = entrenador.completar_fim(
    prefijo="public async Task<ProductDto> GetByIdAsync(int id)\n{\n    ",
    sufijo="\n    return _mapper.Map<ProductDto>(product);\n}",
    max_tokens=50,
)
print(resultado["middle_generado"])
```

### 4. RAG: Crear endpoint REST

```python
comparacion = sistema_rag.comparar_con_sin_rag(
    query="crear endpoint API REST para gestionar pedidos",
    modelo=modelo,
    tokenizador=tokenizador,
)
# Compara generación con contexto RAG vs sin contexto
```

---

## 🖥️ Visualización Interactiva

La visualización HTML/JS incluye 6 tabs:

| Tab | Qué muestra |
|---|---|
| **Resumen** | Estadísticas del modelo, arquitectura, curva de entrenamiento |
| **Tokenización** | Tokens coloreados por categoría con tooltips |
| **Attention Maps** | Heat maps de atención por capa (qué "mira" el modelo) |
| **Generación** | Generación token por token con candidatos alternativos |
| **Flujo RAG** | Pipeline completo: query → FAISS → snippets → contexto |
| **Con vs Sin RAG** | Comparación lado a lado con métricas |

---

## 📚 Para Aprender Más

- **"Attention Is All You Need"** (Vaswani et al., 2017) — Paper original del Transformer
- **"Efficient Transformers for Code"** — Cómo se adaptan los Transformers para código
- **"Retrieval-Augmented Generation"** (Lewis et al., 2020) — Paper original de RAG
- **FAISS Documentation** — https://faiss.ai/
- **The Stack** (HuggingFace) — Dataset real de código para entrenar modelos

---

*Proyecto educativo creado para entender los mecanismos internos de los generadores de código.*
*~3000 líneas de código con comentarios extensos en español.*
