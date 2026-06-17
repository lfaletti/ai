"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MINI TRANSFORMER EDUCATIVO EN PYTORCH                    ║
║                                                                            ║
║  Una implementación didáctica de un Transformer decoder-only para          ║
║  generación de texto, similar en espíritu a GPT.                           ║
║                                                                            ║
║  Cada componente incluye explicaciones detalladas de:                      ║
║    - QUÉ hace                                                              ║
║    - POR QUÉ es necesario                                                  ║
║    - CÓMO fluyen las dimensiones de los tensores                           ║
║    - La INTUICIÓN detrás de cada operación                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA GENERAL (Decoder-Only Transformer):
================================================

  Texto de entrada: "El gato se"
       │
       ▼
  ┌─────────────┐
  │  Tokenizer   │  → Convierte texto a números (IDs de tokens)
  └──────┬──────┘
         │  [token_ids]: (batch_size, seq_len)
         ▼
  ┌─────────────────┐
  │ Token Embedding  │  → Cada token se convierte en un vector denso
  └──────┬──────────┘
         │  (batch_size, seq_len, d_model)
         ▼
  ┌──────────────────────┐
  │ Positional Encoding   │  → Añade información de posición
  └──────┬───────────────┘
         │  (batch_size, seq_len, d_model)
         ▼
  ┌──────────────────────┐
  │  Decoder Layer × N    │  → N capas de transformación
  │  ┌──────────────────┐ │
  │  │ Multi-Head Attn  │ │  → "¿A qué tokens debo prestar atención?"
  │  │ + LayerNorm      │ │
  │  │ + Residual       │ │
  │  ├──────────────────┤ │
  │  │ Feed-Forward Net │ │  → Procesamiento no lineal
  │  │ + LayerNorm      │ │
  │  │ + Residual       │ │
  │  └──────────────────┘ │
  └──────┬───────────────┘
         │  (batch_size, seq_len, d_model)
         ▼
  ┌─────────────────┐
  │ Linear + Softmax │  → Probabilidades del siguiente token
  └──────┬──────────┘
         │  (batch_size, seq_len, vocab_size)
         ▼
  Predicción: "sentó" (token más probable)
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


# =============================================================================
# COMPONENTE 1: POSITIONAL ENCODING (Codificación Posicional)
# =============================================================================
#
# ¿POR QUÉ SE NECESITA?
# ─────────────────────
# A diferencia de las RNNs que procesan tokens uno por uno (y por tanto
# "saben" el orden implícitamente), el Transformer procesa TODOS los tokens
# en paralelo. Esto es genial para velocidad, pero significa que sin
# Positional Encoding, la frase "el gato persigue al ratón" sería
# idéntica a "el ratón persigue al gato" para el modelo.
#
# ¿CÓMO FUNCIONA?
# ────────────────
# Usamos funciones seno y coseno de diferentes frecuencias para crear
# un "fingerprint" único para cada posición. La fórmula es:
#
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# Donde:
#   - pos = posición del token en la secuencia (0, 1, 2, ...)
#   - i = índice de la dimensión del embedding (0, 1, 2, ..., d_model/2)
#   - d_model = dimensión del modelo
#
# INTUICIÓN: Piensa en esto como un "reloj" con muchas manecillas que giran
# a diferentes velocidades. Cada posición tiene una combinación única de
# valores, como las horas en un reloj analógico (la manecilla de horas,
# minutos y segundos juntas te dan una "firma" única del momento).
#
# VENTAJA de sin/cos sobre embeddings aprendidos:
#   - Puede generalizar a secuencias más largas que las vistas en entrenamiento
#   - Las relaciones de distancia relativa se preservan (PE[pos+k] se puede
#     expresar como una transformación lineal de PE[pos])

class PositionalEncoding(nn.Module):
    """
    Añade información posicional a los embeddings de tokens.

    Sin esto, "Juan ama a María" y "María ama a Juan" serían
    indistinguibles para el modelo.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: Dimensión de los embeddings del modelo (ej: 128, 256, 512)
            max_seq_len: Longitud máxima de secuencia soportada
            dropout: Probabilidad de dropout para regularización
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creamos una matriz de codificación posicional
        # Dimensiones: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)

        # Vector de posiciones: [0, 1, 2, ..., max_seq_len-1]
        # Dimensiones: (max_seq_len, 1) - columna vertical
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Factor de escala para las frecuencias
        # Este término crea frecuencias que van desde alta (dimensiones bajas)
        # hasta baja (dimensiones altas), cubriendo un amplio rango
        # Dimensiones: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Dimensiones pares (0, 2, 4, ...): usamos seno
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_seq_len, d_model/2)

        # Dimensiones impares (1, 3, 5, ...): usamos coseno
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_seq_len, d_model/2)

        # Añadimos dimensión de batch: (1, max_seq_len, d_model)
        # register_buffer: se guarda con el modelo pero NO se entrena
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de embeddings de tokens
               Dimensiones: (batch_size, seq_len, d_model)

        Returns:
            Embeddings + codificación posicional
            Dimensiones: (batch_size, seq_len, d_model)  [no cambian]
        """
        # Sumamos (no concatenamos) la codificación posicional a los embeddings
        # Solo tomamos las posiciones que necesitamos (:x.size(1))
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# COMPONENTE 2: SCALED DOT-PRODUCT ATTENTION (Atención Escalada)
# =============================================================================
#
# ¿QUÉ ES LA ATENCIÓN?
# ─────────────────────
# Es el mecanismo que permite a cada token "mirar" a otros tokens para
# decidir cuánta importancia darle a cada uno. Por ejemplo, en:
#   "El gato que estaba en el tejado se cayó"
# Cuando el modelo procesa "se cayó", necesita saber que el sujeto es
# "el gato" (no "el tejado"), y la atención le permite hacer esa conexión.
#
# ¿CÓMO FUNCIONA? (Q, K, V)
# ──────────────────────────
# Imagina una biblioteca:
#   - Query (Q) = "¿Qué estoy buscando?" → La pregunta que hace cada token
#   - Key (K)   = "¿Qué información tengo?" → La etiqueta de cada token
#   - Value (V) = "¿Cuál es mi contenido?" → La información real de cada token
#
# El proceso:
#   1. Cada token genera su Q, K y V (mediante multiplicación por matrices)
#   2. Se calcula qué tan compatible es cada Q con cada K (producto punto)
#   3. Se normaliza con softmax para obtener pesos de atención (suman 1)
#   4. Se usa esos pesos para hacer un promedio ponderado de los V
#
# Fórmula: Attention(Q, K, V) = softmax(QK^T / √d_k) × V
#
# ¿POR QUÉ DIVIDIR POR √d_k?
# ───────────────────────────
# Sin esta escala, cuando d_k es grande, los productos punto pueden ser
# números muy grandes, lo que hace que softmax produzca distribuciones
# casi one-hot (un valor ~1, todos los demás ~0). Esto causa gradientes
# muy pequeños y dificulta el entrenamiento. Dividir por √d_k mantiene
# la varianza estable.

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
    return_attention_weights: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Calcula la atención escalada por producto punto.

    Args:
        query:  (batch, num_heads, seq_len, d_k) - "¿Qué busco?"
        key:    (batch, num_heads, seq_len, d_k) - "¿Qué etiqueta tengo?"
        value:  (batch, num_heads, seq_len, d_k) - "¿Cuál es mi contenido?"
        mask:   Máscara para evitar atención a tokens futuros (causal mask)
        dropout: Capa de dropout opcional
        return_attention_weights: Si True, retorna también los pesos de atención

    Returns:
        output: (batch, num_heads, seq_len, d_k) - Resultado de la atención
        attention_weights: (batch, num_heads, seq_len, seq_len) - Pesos [opcional]
    """
    d_k = query.size(-1)  # Dimensión de cada cabeza

    # ╔═══════════════════════════════════════════════════╗
    # ║ PASO 1: Calcular scores de compatibilidad        ║
    # ╚═══════════════════════════════════════════════════╝
    # query: (batch, heads, seq_len, d_k)
    # key^T: (batch, heads, d_k, seq_len)  ← transponemos las 2 últimas dims
    # scores: (batch, heads, seq_len, seq_len)  ← cada token vs cada token
    #
    # scores[i][j] = "¿Cuánto debería el token i prestar atención al token j?"
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # ╔═══════════════════════════════════════════════════╗
    # ║ PASO 2: Aplicar máscara causal (si existe)       ║
    # ╚═══════════════════════════════════════════════════╝
    # En generación de texto, un token NO puede atender a tokens futuros.
    # La máscara pone -infinito en las posiciones futuras, que softmax
    # convierte en 0 (e^(-inf) = 0).
    #
    # Ejemplo para secuencia de 4 tokens:
    #   scores antes de máscara:     después de máscara:
    #   [0.5  0.3  0.8  0.1]        [0.5  -inf  -inf  -inf]  token 0 solo ve a sí mismo
    #   [0.2  0.7  0.4  0.6]        [0.2   0.7  -inf  -inf]  token 1 ve tokens 0,1
    #   [0.9  0.1  0.5  0.3]        [0.9   0.1   0.5  -inf]  token 2 ve tokens 0,1,2
    #   [0.4  0.8  0.2  0.7]        [0.4   0.8   0.2   0.7]  token 3 ve todos
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # ╔═══════════════════════════════════════════════════╗
    # ║ PASO 3: Softmax → pesos de atención              ║
    # ╚═══════════════════════════════════════════════════╝
    # Convertimos scores en probabilidades (suman 1 por cada fila)
    # attention_weights: (batch, heads, seq_len, seq_len)
    attention_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # ╔═══════════════════════════════════════════════════╗
    # ║ PASO 4: Promedio ponderado de Values              ║
    # ╚═══════════════════════════════════════════════════╝
    # attention_weights: (batch, heads, seq_len, seq_len)
    # value:             (batch, heads, seq_len, d_k)
    # output:            (batch, heads, seq_len, d_k)
    #
    # Para cada token, su output es un promedio ponderado de todos los
    # valores, donde los pesos son los attention_weights.
    output = torch.matmul(attention_weights, value)

    if return_attention_weights:
        return output, attention_weights
    return output, None


# =============================================================================
# COMPONENTE 3: MULTI-HEAD ATTENTION (Atención Multi-Cabeza)
# =============================================================================
#
# ¿POR QUÉ MÚLTIPLES CABEZAS?
# ────────────────────────────
# Una sola cabeza de atención solo puede aprender UN tipo de relación.
# Con múltiples cabezas, el modelo puede atender a diferentes aspectos
# simultáneamente. Por ejemplo:
#
#   Cabeza 1: Relaciones sintácticas (sujeto ↔ verbo)
#   Cabeza 2: Relaciones semánticas (sinónimos, contexto)
#   Cabeza 3: Relaciones posicionales (tokens cercanos)
#   Cabeza 4: Dependencias a larga distancia
#
# Es como tener múltiples "expertos" que miran la misma secuencia desde
# diferentes ángulos y luego combinan sus conclusiones.
#
# IMPLEMENTACIÓN:
# En lugar de hacer h atenciones separadas con d_model dimensiones cada una,
# dividimos d_model en h cabezas de d_k = d_model/h dimensiones cada una.
# Esto es computacionalmente equivalente pero más eficiente.

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: múltiples cabezas de atención en paralelo.

    Cada cabeza puede aprender a atender a diferentes tipos de relaciones
    entre tokens, enriqueciendo la representación del modelo.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimensión total del modelo (ej: 128)
            num_heads: Número de cabezas de atención (ej: 4)
                       d_model debe ser divisible por num_heads
            dropout: Probabilidad de dropout
        """
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensión por cabeza

        # ┌─────────────────────────────────────────────────────────┐
        # │ Matrices de proyección para Q, K, V                     │
        # │                                                         │
        # │ Cada una transforma el input de (d_model) a (d_model)   │
        # │ que luego se divide en num_heads cabezas de (d_k) cada  │
        # │ una.                                                    │
        # │                                                         │
        # │ W_q, W_k, W_v: (d_model, d_model)                      │
        # └─────────────────────────────────────────────────────────┘
        self.W_q = nn.Linear(d_model, d_model)  # Proyección de Query
        self.W_k = nn.Linear(d_model, d_model)  # Proyección de Key
        self.W_v = nn.Linear(d_model, d_model)  # Proyección de Value

        # Proyección de salida: combina las cabezas de vuelta
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Para almacenar los pesos de atención (útil para visualización)
        self._attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Máscara causal (batch_size, 1, seq_len, seq_len)
            return_attention: Si retornar pesos de atención

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len) [opcional]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 1: Proyectar a Q, K, V                      ║
        # ╚═══════════════════════════════════════════════════╝
        # x: (batch, seq_len, d_model) → Q/K/V: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 2: Dividir en múltiples cabezas             ║
        # ╚═══════════════════════════════════════════════════╝
        # Reshape: (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        # Transpose: → (batch, num_heads, seq_len, d_k)
        #
        # Esto permite que cada cabeza procese su porción del embedding
        # de forma independiente.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Ahora: Q, K, V tienen dimensiones (batch, num_heads, seq_len, d_k)

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 3: Calcular atención para todas las cabezas ║
        # ╚═══════════════════════════════════════════════════╝
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout,
            return_attention_weights=return_attention
        )
        # attn_output: (batch, num_heads, seq_len, d_k)
        # attn_weights: (batch, num_heads, seq_len, seq_len)

        if attn_weights is not None:
            self._attention_weights = attn_weights.detach()

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 4: Concatenar cabezas y proyectar           ║
        # ╚═══════════════════════════════════════════════════╝
        # Transpose: (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        # Reshape:   → (batch, seq_len, d_model)  [d_model = num_heads × d_k]
        #
        # Esto "pega" las salidas de todas las cabezas de vuelta
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Proyección final: mezcla la información de todas las cabezas
        output = self.W_o(attn_output)  # (batch, seq_len, d_model)

        return output, attn_weights


# =============================================================================
# COMPONENTE 4: FEED-FORWARD NETWORK (Red de Alimentación Directa)
# =============================================================================
#
# ¿POR QUÉ SE NECESITA?
# ─────────────────────
# La atención es excelente para MEZCLAR información entre tokens, pero
# solo hace operaciones lineales (productos punto y promedios ponderados).
# La Feed-Forward Network añade NO-LINEALIDAD, permitiendo al modelo
# aprender transformaciones complejas de cada token individualmente.
#
# Es como si la atención dijera "estos tokens son relevantes entre sí"
# y la FFN dijera "ahora déjame pensar profundamente sobre qué hacer
# con esta información combinada".
#
# ESTRUCTURA:
#   Linear(d_model → d_ff) → ReLU/GELU → Linear(d_ff → d_model)
#
# Típicamente d_ff = 4 × d_model (expansión y luego compresión),
# creando un "cuello de botella" que fuerza al modelo a aprender
# representaciones compactas y útiles.

class FeedForward(nn.Module):
    """
    Red Feed-Forward de dos capas con activación no lineal.

    Procesa cada posición (token) de forma independiente, añadiendo
    capacidad de transformación no lineal al modelo.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimensión del modelo (entrada y salida)
            d_ff: Dimensión de la capa intermedia (típicamente 4 × d_model)
            dropout: Probabilidad de dropout
        """
        super().__init__()

        # Capa 1: Expansión  (d_model → d_ff)
        # Ejemplo: 128 → 512  (× 4 más dimensiones para "pensar")
        self.linear1 = nn.Linear(d_model, d_ff)

        # Capa 2: Compresión  (d_ff → d_model)
        # Ejemplo: 512 → 128  (volvemos a la dimensión original)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

        # GELU (Gaussian Error Linear Unit) es la activación preferida
        # en Transformers modernos. Es similar a ReLU pero más suave,
        # lo que ayuda al entrenamiento.
        # GELU(x) ≈ x × Φ(x), donde Φ es la CDF de la normal estándar
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)  [mismas dimensiones]

        Flujo de dimensiones:
            (batch, seq_len, d_model=128)
                → linear1 → (batch, seq_len, d_ff=512)   [expansión]
                → GELU    → (batch, seq_len, d_ff=512)   [no-linealidad]
                → dropout → (batch, seq_len, d_ff=512)
                → linear2 → (batch, seq_len, d_model=128) [compresión]
        """
        x = self.linear1(x)      # Expansión
        x = self.activation(x)   # No-linealidad
        x = self.dropout(x)
        x = self.linear2(x)      # Compresión de vuelta
        return x


# =============================================================================
# COMPONENTE 5: DECODER LAYER (Capa del Decodificador)
# =============================================================================
#
# ¿QUÉ SON LAS CONEXIONES RESIDUALES?
# ────────────────────────────────────
# Cada sub-capa (atención y FFN) tiene una "autopista" que permite al
# gradiente fluir directamente a través de la red durante el entrenamiento:
#
#   output = LayerNorm(x + SubLayer(x))
#
# Sin conexiones residuales, apilar muchas capas causaría el problema del
# "gradiente que se desvanece" y la red no aprendería. Con ellas, si una
# capa no aporta nada útil, simplemente deja pasar x sin modificar.
#
# ¿QUÉ ES LAYER NORMALIZATION?
# ────────────────────────────
# Normaliza los valores dentro de cada ejemplo (no entre ejemplos del batch).
# Para cada token: centra los valores a media=0, varianza=1, y luego
# los escala con parámetros aprendidos.
#
# Esto estabiliza el entrenamiento al evitar que los valores crezcan o
# se reduzcan demasiado a medida que pasan por muchas capas.
#
# NOTA: Usamos "Pre-LayerNorm" (normalizar ANTES de la subcapa),
# que es lo que usan GPT-2/3 y la mayoría de modelos modernos, porque
# produce un entrenamiento más estable.

class DecoderLayer(nn.Module):
    """
    Una capa del Decoder Transformer.

    Estructura (Pre-LayerNorm):
        x → LayerNorm → Multi-Head Attention → + residual → 
          → LayerNorm → Feed-Forward → + residual → output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión de la capa intermedia del FFN
            dropout: Probabilidad de dropout
        """
        super().__init__()

        # Sub-capa 1: Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-capa 2: Feed-Forward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer Normalization para cada sub-capa (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout para las conexiones residuales
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Máscara causal
            return_attention: Si retornar pesos de atención

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: Pesos de atención [opcional]
        """
        # ╔═══════════════════════════════════════════════════╗
        # ║ SUB-CAPA 1: Multi-Head Self-Attention             ║
        # ║ + Conexión Residual + Layer Normalization          ║
        # ╚═══════════════════════════════════════════════════╝
        # 1. Normalizamos ANTES de la atención (Pre-LN)
        x_norm = self.norm1(x)
        # 2. Calculamos la atención
        attn_output, attn_weights = self.self_attention(
            x_norm, mask=mask, return_attention=return_attention
        )
        # 3. Conexión residual: sumamos el input original
        #    Esto permite que el gradiente fluya directamente
        x = x + self.dropout(attn_output)

        # ╔═══════════════════════════════════════════════════╗
        # ║ SUB-CAPA 2: Feed-Forward Network                  ║
        # ║ + Conexión Residual + Layer Normalization          ║
        # ╚═══════════════════════════════════════════════════╝
        # 1. Normalizamos ANTES del FFN
        x_norm = self.norm2(x)
        # 2. Pasamos por la red feed-forward
        ff_output = self.feed_forward(x_norm)
        # 3. Otra conexión residual
        x = x + self.dropout(ff_output)

        return x, attn_weights


# =============================================================================
# COMPONENTE 6: MINI TRANSFORMER (Modelo Completo)
# =============================================================================
#
# Este es el modelo completo que integra todos los componentes anteriores.
# Es un "decoder-only" Transformer, similar a GPT:
#
#   1. Tokenización (simple, por caracteres en nuestro caso)
#   2. Token Embedding + Positional Encoding
#   3. N capas de DecoderLayer
#   4. Layer Norm final
#   5. Proyección a vocabulario (logits)
#
# El entrenamiento usa "teacher forcing": dada una secuencia, el modelo
# predice el siguiente token en cada posición, y comparamos con la
# secuencia desplazada un token a la derecha.

class MiniTransformer(nn.Module):
    """
    Mini Transformer Decoder-Only para generación de texto.

    Un modelo educativo que implementa la arquitectura completa de
    un Transformer generativo, similar en espíritu a GPT.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        token_to_id: Optional[Dict[str, int]] = None,
        id_to_token: Optional[Dict[int, str]] = None
    ):
        """
        Args:
            vocab_size: Tamaño del vocabulario (número total de tokens únicos)
            d_model: Dimensión de los embeddings y del modelo (128)
            num_heads: Número de cabezas de atención (4)
            num_layers: Número de capas del decoder (4)
            d_ff: Dimensión de la capa intermedia del FFN (512)
            max_seq_len: Longitud máxima de secuencia (256)
            dropout: Probabilidad de dropout (0.1)
            token_to_id: Diccionario token → ID (para serialización)
            id_to_token: Diccionario ID → token (para serialización)
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_to_id = token_to_id or {}
        self.id_to_token = id_to_token or {}

        # ┌─────────────────────────────────────────────────────────┐
        # │ Token Embedding                                         │
        # │                                                         │
        # │ Convierte IDs de tokens en vectores densos.             │
        # │ Cada token tiene su propio vector de d_model dimensiones│
        # │ que se APRENDE durante el entrenamiento.                │
        # │                                                         │
        # │ Ejemplo: token "gato" (ID=42) → [0.23, -0.15, ..., 0.8]│
        # │ (un vector de 128 dimensiones que captura el            │
        # │  "significado" de "gato" en el espacio del modelo)      │
        # └─────────────────────────────────────────────────────────┘
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Codificación posicional (componente 1)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Capas del decoder (componente 5 × num_layers)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Layer Norm final (estabiliza la salida antes de la proyección)
        self.final_norm = nn.LayerNorm(d_model)

        # ┌─────────────────────────────────────────────────────────┐
        # │ Capa de Salida (Language Model Head)                    │
        # │                                                         │
        # │ Proyecta de d_model dimensiones a vocab_size para       │
        # │ obtener los logits (puntuaciones sin normalizar) de     │
        # │ cada token del vocabulario.                             │
        # │                                                         │
        # │ (batch, seq_len, d_model) → (batch, seq_len, vocab_size)│
        # └─────────────────────────────────────────────────────────┘
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Compartir pesos entre embedding y output (weight tying)
        # Esta es una técnica común que:
        # 1. Reduce el número de parámetros
        # 2. Mejora el rendimiento (tokens con embeddings similares
        #    tendrán probabilidades de salida similares)
        if d_model == self.token_embedding.embedding_dim:
            self.output_projection.weight = self.token_embedding.weight

        # Información del modelo para serialización
        self._config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'd_ff': d_ff,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
        }

        # Inicialización de pesos (Xavier/Glorot para estabilidad)
        self._init_weights()

    def _init_weights(self):
        """
        Inicializa los pesos del modelo con la estrategia Xavier.

        Una buena inicialización es crucial: si los pesos iniciales son
        demasiado grandes, los gradientes explotan; si son demasiado
        pequeños, desaparecen. Xavier mantiene la varianza estable.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Genera una máscara causal (triangular inferior).

        Esto asegura que el token en posición i solo puede atender a
        tokens en posiciones ≤ i (no puede "ver el futuro").

        Para seq_len=4:
            [[1, 0, 0, 0],    token 0: solo ve a sí mismo
             [1, 1, 0, 0],    token 1: ve tokens 0, 1
             [1, 1, 1, 0],    token 2: ve tokens 0, 1, 2
             [1, 1, 1, 1]]    token 3: ve todos

        Returns:
            mask: (1, 1, seq_len, seq_len) - 1 donde se permite atención, 0 donde no
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # Añadir dims de batch y head

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass completo del modelo.

        Args:
            x: Token IDs (batch_size, seq_len) - enteros
            return_attention: Si retornar pesos de atención de cada capa

        Returns:
            logits: (batch_size, seq_len, vocab_size) - puntuaciones sin normalizar
            attention_weights: Lista de (batch, heads, seq_len, seq_len) por capa [opcional]
        """
        seq_len = x.size(1)
        device = x.device

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 1: Token Embedding                           ║
        # ╚═══════════════════════════════════════════════════╝
        # x: (batch, seq_len) de enteros
        # → embeddings: (batch, seq_len, d_model) de floats
        #
        # Multiplicamos por √d_model para escalar los embeddings.
        # Esto es una convención del paper original que balancea
        # las magnitudes entre embeddings y positional encoding.
        embeddings = self.token_embedding(x) * math.sqrt(self.d_model)

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 2: Añadir Positional Encoding                ║
        # ╚═══════════════════════════════════════════════════╝
        # embeddings: (batch, seq_len, d_model)
        # → con posición: (batch, seq_len, d_model)
        h = self.positional_encoding(embeddings)

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 3: Crear máscara causal                      ║
        # ╚═══════════════════════════════════════════════════╝
        mask = self._generate_causal_mask(seq_len, device)

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 4: Pasar por N capas del decoder              ║
        # ╚═══════════════════════════════════════════════════╝
        all_attention_weights = []
        for layer in self.layers:
            h, attn_w = layer(h, mask=mask, return_attention=return_attention)
            if attn_w is not None:
                all_attention_weights.append(attn_w)

        # ╔═══════════════════════════════════════════════════╗
        # ║ PASO 5: Layer Norm final + Proyección a vocab     ║
        # ╚═══════════════════════════════════════════════════╝
        # h: (batch, seq_len, d_model)
        h = self.final_norm(h)
        # logits: (batch, seq_len, vocab_size)
        logits = self.output_projection(h)

        if return_attention:
            return logits, all_attention_weights
        return logits, None

    # =========================================================================
    # GENERACIÓN DE TEXTO
    # =========================================================================

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Genera texto token por token usando el modelo entrenado.

        El proceso de generación es AUTOREGRESIVO:
        1. Dada una secuencia inicial (prompt), predecimos el siguiente token
        2. Añadimos ese token al final de la secuencia
        3. Repetimos hasta alcanzar max_new_tokens

        Args:
            prompt_ids: (1, prompt_len) - IDs de tokens iniciales
            max_new_tokens: Cuántos tokens nuevos generar
            temperature: Controla la "creatividad" (0.1=conservador, 1.5=creativo)
                        Valores bajos → texto más predecible y repetitivo
                        Valores altos → texto más variado pero potencialmente incoherente
            top_k: Si > 0, solo considerar los k tokens más probables
            top_p: Nucleus sampling - solo considerar tokens cuya probabilidad
                   acumulada sea ≤ top_p (0.9 = considerar el 90% de masa)

        Returns:
            generated_ids: (1, prompt_len + max_new_tokens) - Secuencia completa
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncar si excede la longitud máxima
            context = generated[:, -self.max_seq_len:]

            # Forward pass: obtener logits para el último token
            logits, _ = self.forward(context)
            # Solo nos interesa la predicción del ÚLTIMO token
            # logits[:, -1, :] → (batch=1, vocab_size)
            next_token_logits = logits[:, -1, :]

            # ┌──────────────────────────────────────────────────┐
            # │ TEMPERATURA                                       │
            # │ Dividir logits por temperatura antes de softmax:  │
            # │ - T < 1: distribución más "puntiaguda" (seguro)  │
            # │ - T = 1: distribución original                   │
            # │ - T > 1: distribución más "plana" (diverso)      │
            # └──────────────────────────────────────────────────┘
            next_token_logits = next_token_logits / max(temperature, 1e-8)

            # ┌──────────────────────────────────────────────────┐
            # │ TOP-K SAMPLING                                    │
            # │ Solo considerar los K tokens más probables.       │
            # │ Todos los demás se ponen a -inf (prob = 0).       │
            # └──────────────────────────────────────────────────┘
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                # Encontrar el valor del k-ésimo logit más alto
                kth_values = torch.topk(next_token_logits, top_k_val).values[:, -1:]
                # Filtrar todos los que están por debajo
                next_token_logits[next_token_logits < kth_values] = float('-inf')

            # ┌──────────────────────────────────────────────────┐
            # │ TOP-P (NUCLEUS) SAMPLING                          │
            # │ Solo considerar el conjunto mínimo de tokens cuya │
            # │ probabilidad acumulada alcance p.                 │
            # │                                                   │
            # │ Ejemplo con top_p=0.9:                            │
            # │ Si "gato"=0.4, "perro"=0.3, "pez"=0.2, "rana"=0.1│
            # │ Acumulada: 0.4, 0.7, 0.9, 1.0                    │
            # │ Solo consideramos {"gato", "perro", "pez"}        │
            # └──────────────────────────────────────────────────┘
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Encontrar dónde la probabilidad acumulada excede top_p
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                # Restaurar el orden original
                next_token_logits = sorted_logits.scatter(
                    1, sorted_indices, sorted_logits
                )

            # Convertir logits a probabilidades y samplear
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Añadir el token generado a la secuencia
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def generate_with_trace(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 10,
        temperature: float = 0.8,
    ) -> Dict:
        """
        Genera texto guardando información detallada de cada paso
        para visualización educativa.

        Returns:
            Dict con toda la información del proceso de generación.
        """
        self.eval()
        generated = prompt_ids.clone()
        trace = {
            'steps': [],
            'prompt_ids': prompt_ids[0].tolist(),
            'config': self._config,
        }

        for step in range(max_new_tokens):
            context = generated[:, -self.max_seq_len:]

            # Forward con atención
            logits, attention_weights = self.forward(context, return_attention=True)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(next_logits, dim=-1)

            # Top-5 predicciones
            top_probs, top_indices = torch.topk(probs[0], min(5, probs.size(-1)))

            # Samplear
            next_token = torch.multinomial(probs, num_samples=1)

            step_info = {
                'step': step,
                'context_ids': context[0].tolist(),
                'top_predictions': [
                    {'token_id': idx.item(), 'probability': p.item()}
                    for p, idx in zip(top_probs, top_indices)
                ],
                'chosen_token_id': next_token.item(),
                'chosen_probability': probs[0, next_token.item()].item(),
                'attention_weights': [
                    aw[0].cpu().numpy().tolist()  # (heads, seq, seq)
                    for aw in attention_weights
                ] if attention_weights else [],
            }
            trace['steps'].append(step_info)

            generated = torch.cat([generated, next_token], dim=1)

        trace['generated_ids'] = generated[0].tolist()
        return trace

    # =========================================================================
    # GUARDAR Y CARGAR MODELO
    # =========================================================================

    def save(self, path: str):
        """
        Guarda el modelo completo (pesos + configuración + vocabulario).

        Args:
            path: Ruta del archivo .pt donde guardar
        """
        checkpoint = {
            'config': self._config,
            'state_dict': self.state_dict(),
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
        }
        torch.save(checkpoint, path)
        print(f"✅ Modelo guardado en: {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'MiniTransformer':
        """
        Carga un modelo guardado previamente.

        Args:
            path: Ruta del archivo .pt
            device: Dispositivo donde cargar ('cpu' o 'cuda')

        Returns:
            Modelo cargado y listo para usar
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']

        model = cls(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
            token_to_id=checkpoint.get('token_to_id', {}),
            id_to_token=checkpoint.get('id_to_token', {}),
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"✅ Modelo cargado desde: {path}")
        print(f"   Configuración: {config}")
        return model

    def count_parameters(self) -> int:
        """Cuenta el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        params = self.count_parameters()
        return (
            f"MiniTransformer(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self._config['d_model']},\n"
            f"  num_heads={self._config['num_heads']},\n"
            f"  num_layers={self._config['num_layers']},\n"
            f"  d_ff={self._config['d_ff']},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  total_parameters={params:,}\n"
            f")"
        )


# =============================================================================
# TOKENIZER SIMPLE (Por Caracteres)
# =============================================================================
#
# En producción se usan tokenizers como BPE (GPT) o SentencePiece (LLaMA),
# pero para fines educativos usamos un tokenizer por caracteres que es
# más simple de entender y depurar.
#
# Cada carácter único en el texto es un token.
# Ejemplo: "hola" → [h, o, l, a] → [23, 45, 38, 12]

class CharTokenizer:
    """
    Tokenizer simple a nivel de carácter.

    Convierte texto ↔ secuencias de IDs enteros.
    Útil para entender el concepto sin la complejidad de BPE/WordPiece.
    """

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> 'CharTokenizer':
        """
        Construye el vocabulario a partir de un texto.

        Args:
            text: Texto de entrenamiento

        Returns:
            self (para encadenar llamadas)
        """
        # Encontrar todos los caracteres únicos y ordenarlos
        chars = sorted(set(text))
        self.token_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_token = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"📝 Vocabulario creado: {self.vocab_size} tokens únicos")
        print(f"   Caracteres: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")
        return self

    def encode(self, text: str) -> List[int]:
        """Convierte texto → lista de IDs."""
        return [self.token_to_id[ch] for ch in text if ch in self.token_to_id]

    def decode(self, ids: List[int]) -> str:
        """Convierte lista de IDs → texto."""
        return ''.join(self.id_to_token.get(i, '?') for i in ids)

    def to_dict(self) -> Dict:
        """Serializa el tokenizer a diccionario."""
        return {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'vocab_size': self.vocab_size,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'CharTokenizer':
        """Carga el tokenizer desde un diccionario."""
        tok = cls()
        tok.token_to_id = d['token_to_id']
        tok.id_to_token = {int(k): v for k, v in d['id_to_token'].items()}
        tok.vocab_size = d['vocab_size']
        return tok


# =============================================================================
# DATASET PARA ENTRENAMIENTO
# =============================================================================

class TextDataset(torch.utils.data.Dataset):
    """
    Dataset que crea pares (input, target) para entrenamiento.

    De un texto largo, extrae ventanas deslizantes de tamaño seq_len.
    El target es el input desplazado un token a la derecha.

    Ejemplo con seq_len=5:
        Texto: "El gato duerme"
        Input:  "El ga"  → Target: "l gat"
        Input:  "l gat"  → Target: " gato"
        ...

    Esto enseña al modelo: dado un contexto, ¿cuál es el siguiente token?
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int = 64):
        self.seq_len = seq_len
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        # Número de ventanas que podemos extraer
        self.num_samples = max(0, len(self.data) - seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: tokens[idx : idx+seq_len]
        # Target: tokens[idx+1 : idx+seq_len+1]  (desplazado 1 a la derecha)
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


if __name__ == "__main__":
    # Quick sanity check
    print("=" * 60)
    print("Mini Transformer - Verificación rápida")
    print("=" * 60)

    model = MiniTransformer(vocab_size=50, d_model=64, num_heads=4, num_layers=2, d_ff=256)
    print(model)

    # Test forward pass
    dummy_input = torch.randint(0, 50, (2, 16))
    logits, _ = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"✅ Forward pass exitoso!")
