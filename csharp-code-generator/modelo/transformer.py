"""
============================================================================
modelo/transformer.py — Transformer para Generación de Código C#
============================================================================

PROPÓSITO EDUCATIVO:
    Implementación COMPLETA de un Transformer decoder-only de 4 capas
    para generación de código C#. Cada componente está documentado
    extensamente en español explicando:

    1. POR QUÉ cada componente existe
    2. CÓMO se diferencia de un Transformer para texto natural
    3. QUÉ aprende el modelo sobre la estructura del código C#

ARQUITECTURA:
    Usamos un modelo decoder-only (como GPT) porque la generación de
    código es una tarea AUTOREGRESIVA: el modelo genera un token a la
    vez, condicionado en todos los tokens anteriores.

    Componentes:
    ┌─────────────────────────────────────────┐
    │  Embedding de Tokens (tam_vocab → d_model) │
    │  + Embedding Posicional                     │
    ├─────────────────────────────────────────┤
    │  Capa Transformer × 4                       │
    │  ├─ Multi-Head Self-Attention (causal)      │
    │  ├─ Add & Norm                              │
    │  ├─ Feed-Forward Network                    │
    │  └─ Add & Norm                              │
    ├─────────────────────────────────────────┤
    │  Capa de Salida (d_model → tam_vocab)       │
    │  → Probabilidad de cada token               │
    └─────────────────────────────────────────┘

DIFERENCIAS CLAVE VS MODELO DE TEXTO NATURAL:
    1. VOCABULARIO ESPECIALIZADO: Tokens de código C# (keywords, operadores)
    2. CONTEXTO MÁS LARGO: El código tiene dependencias a larga distancia
       (una llave "{" puede cerrarse 100 líneas después)
    3. TOKENS FIM: Soporte para Fill-in-the-Middle
    4. ATTENTION PATTERNS: En código, la atención se concentra en:
       - Declaraciones de tipo (para coherencia de tipos)
       - Llaves de apertura (para saber qué bloque estamos cerrando)
       - Nombre de la clase/método (para coherencia de nombres)
============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


# ============================================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================================

@dataclass
class ConfiguracionTransformer:
    """
    Configuración del Transformer para generación de código C#.

    NOTA EDUCATIVA — ELECCIÓN DE HIPERPARÁMETROS:

    tam_vocabulario (8000):
        Más pequeño que GPT (50k) porque nos enfocamos solo en C#.
        Un vocabulario más pequeño = embeddings más pequeños = modelo
        más rápido de entrenar. Suficiente para un prototipo educativo.

    d_modelo (256):
        Dimensión de los embeddings. Cada token se representa como
        un vector de 256 números. GPT-2 usa 768, GPT-3 usa 12288.
        256 es suficiente para aprender patrones básicos de C#.

    num_cabezas (8):
        Número de "cabezas" de atención. Cada cabeza puede aprender
        a enfocarse en diferentes aspectos del código:
        - Cabeza 1: relación entre "{" y "}"
        - Cabeza 2: relación entre tipo y variable
        - Cabeza 3: relación entre "async" y "await"
        - etc.

    num_capas (4):
        Número de capas del Transformer. Cada capa refina la
        representación. 4 capas son suficientes para patrones simples.
        GPT-2 tiene 12 capas, GPT-4 tiene ~100+.

    dim_ff (1024):
        Dimensión interna del Feed-Forward Network. Típicamente 4x d_modelo.
        Aquí es donde se almacena el "conocimiento" del modelo.

    max_longitud (512):
        Longitud máxima de secuencia. El código C# puede ser largo,
        pero 512 tokens cubren la mayoría de funciones/métodos individuales.
        Copilot usa ~8000 tokens para ver más contexto.

    dropout (0.1):
        Probabilidad de "apagar" neuronas durante entrenamiento.
        Previene overfitting (que el modelo memorice el dataset
        en lugar de aprender patrones generales).
    """
    tam_vocabulario: int = 8000
    d_modelo: int = 256
    num_cabezas: int = 8
    num_capas: int = 4
    dim_ff: int = 1024
    max_longitud: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    # Activar almacenamiento de pesos de atención para visualización
    guardar_atencion: bool = False


# ============================================================================
# COMPONENTES DEL TRANSFORMER
# ============================================================================

class EmbeddingConPosicion(nn.Module):
    """
    Embedding de tokens + Embedding posicional.

    NOTA EDUCATIVA — ¿POR QUÉ EMBEDDINGS?
        Los modelos neuronales no pueden procesar texto directamente.
        Necesitan representaciones numéricas (vectores).

        Un embedding es una tabla de búsqueda: cada token ID tiene
        un vector asociado que el modelo APRENDE durante entrenamiento.

        Ejemplo (simplificado, d_modelo=4):
        "public" → [0.2, -0.5, 0.8, 0.1]
        "class"  → [0.3, -0.4, 0.7, 0.2]
        "int"    → [0.1, 0.6, -0.3, 0.5]

        Tokens con significado similar tendrán vectores similares.
        Por ejemplo, "int" y "string" estarán CERCA en el espacio
        vectorial porque ambos son tipos de dato.

    ¿POR QUÉ EMBEDDING POSICIONAL?
        El Transformer no tiene noción de ORDEN. Sin información
        posicional, "public class Foo" y "class public Foo"
        serían idénticos para el modelo.

        El embedding posicional agrega información de DÓNDE está
        cada token en la secuencia. Usamos embeddings posicionales
        APRENDIDOS (como GPT) en lugar de sinusoidales.

    PARA CÓDIGO C#:
        La posición es MUY importante porque:
        - Las keywords de acceso (public/private) VAN PRIMERO
        - El tipo de retorno va ANTES del nombre del método
        - La indentación codifica la estructura jerárquica
    """

    def __init__(self, config: ConfiguracionTransformer):
        super().__init__()

        # Embedding de tokens: tam_vocab → d_modelo
        self.embedding_token = nn.Embedding(
            config.tam_vocabulario,
            config.d_modelo,
            padding_idx=config.pad_token_id
        )

        # Embedding posicional: max_longitud → d_modelo
        # Cada posición (0, 1, 2, ..., 511) tiene su propio vector
        self.embedding_posicion = nn.Embedding(
            config.max_longitud,
            config.d_modelo
        )

        # Layer normalization para estabilizar el entrenamiento
        self.norm = nn.LayerNorm(config.d_modelo)
        self.dropout = nn.Dropout(config.dropout)

        # Factor de escala para los embeddings (√d_modelo)
        # Esto evita que los embeddings sean demasiado pequeños
        # comparados con los embeddings posicionales
        self.escala = math.sqrt(config.d_modelo)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convierte IDs de tokens en vectores de embeddings.

        Args:
            token_ids: Tensor de forma (batch, seq_len) con IDs de tokens

        Returns:
            Tensor de forma (batch, seq_len, d_modelo) con embeddings

        FLUJO:
            token_ids → embedding_token → escalar → + embedding_posición → norm → dropout
        """
        batch_size, seq_len = token_ids.shape

        # Crear tensor de posiciones: [0, 1, 2, ..., seq_len-1]
        posiciones = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        # Obtener embeddings de tokens y escalar
        x = self.embedding_token(token_ids) * self.escala

        # Sumar embeddings posicionales
        x = x + self.embedding_posicion(posiciones)

        # Normalizar y aplicar dropout
        x = self.norm(x)
        x = self.dropout(x)

        return x


class AtencionMultiCabeza(nn.Module):
    """
    Multi-Head Self-Attention con máscara causal.

    NOTA EDUCATIVA — SELF-ATTENTION EN CÓDIGO:
        Self-Attention permite que cada token "mire" a todos los
        tokens anteriores para decidir qué generar a continuación.

        En CÓDIGO C#, la atención tiene patrones específicos:

        1. ATENCIÓN A TIPOS:
           Cuando el modelo genera un nombre de variable, presta
           mucha atención al tipo declarado antes:
           "int total = " → el modelo sabe que debe generar un valor int

        2. ATENCIÓN A LLAVES:
           Cuando genera "}", el modelo presta atención a la "{"
           correspondiente para saber qué bloque está cerrando.

        3. ATENCIÓN A FIRMAS:
           Dentro de un método, el modelo presta atención a la firma
           del método para saber qué parámetros están disponibles.

        4. ATENCIÓN A IMPORTS/USINGS:
           El modelo verifica qué namespaces están importados
           para saber qué tipos puede usar.

    MÚLTIPLES CABEZAS:
        Cada cabeza puede aprender un PATRÓN DE ATENCIÓN diferente.
        - Cabeza 1: relación tipo-variable
        - Cabeza 2: relación llave apertura-cierre
        - Cabeza 3: relación async-await
        - Cabeza 4: relación namespace-clase

    MÁSCARA CAUSAL:
        Evita que el token en posición i mire tokens en posición > i.
        Esto es ESENCIAL para generación autoregresiva: el modelo
        no puede "hacer trampa" mirando el futuro.
    """

    def __init__(self, config: ConfiguracionTransformer):
        super().__init__()

        self.d_modelo = config.d_modelo
        self.num_cabezas = config.num_cabezas
        self.d_cabeza = config.d_modelo // config.num_cabezas
        self.guardar_atencion = config.guardar_atencion

        assert config.d_modelo % config.num_cabezas == 0, \
            "d_modelo debe ser divisible por num_cabezas"

        # Proyecciones lineales para Q, K, V
        # Q (Query): "¿Qué estoy buscando?"
        # K (Key):   "¿Qué tengo para ofrecer?"
        # V (Value): "¿Qué información llevo?"
        self.W_q = nn.Linear(config.d_modelo, config.d_modelo)
        self.W_k = nn.Linear(config.d_modelo, config.d_modelo)
        self.W_v = nn.Linear(config.d_modelo, config.d_modelo)

        # Proyección de salida
        self.W_o = nn.Linear(config.d_modelo, config.d_modelo)

        self.dropout = nn.Dropout(config.dropout)

        # Almacenar pesos de atención para visualización
        self.pesos_atencion: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mascara: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcula Multi-Head Self-Attention.

        Args:
            x: Tensor de forma (batch, seq_len, d_modelo)
            mascara: Máscara causal (optional, se crea automáticamente)

        Returns:
            Tensor de forma (batch, seq_len, d_modelo) con contexto

        CÁLCULO PASO A PASO:
        1. Proyectar x a Q, K, V
        2. Dividir en múltiples cabezas
        3. Calcular atención: softmax(QK^T / √d_k) × V
        4. Concatenar cabezas
        5. Proyectar de vuelta
        """
        batch_size, seq_len, _ = x.shape

        # Paso 1: Proyecciones lineales
        Q = self.W_q(x)  # (batch, seq_len, d_modelo)
        K = self.W_k(x)
        V = self.W_v(x)

        # Paso 2: Dividir en cabezas
        # Reshape: (batch, seq_len, d_modelo) → (batch, num_cabezas, seq_len, d_cabeza)
        Q = Q.view(batch_size, seq_len, self.num_cabezas, self.d_cabeza).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_cabezas, self.d_cabeza).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_cabezas, self.d_cabeza).transpose(1, 2)

        # Paso 3: Calcular puntajes de atención
        # QK^T / √d_k → (batch, num_cabezas, seq_len, seq_len)
        puntajes = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_cabeza)

        # Aplicar máscara causal (evitar mirar el futuro)
        if mascara is None:
            # Crear máscara causal triangular inferior
            mascara = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()

        puntajes = puntajes.masked_fill(mascara.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax para obtener probabilidades de atención
        pesos = F.softmax(puntajes, dim=-1)
        pesos = self.dropout(pesos)

        # Guardar pesos para visualización (solo si está activado)
        if self.guardar_atencion:
            self.pesos_atencion = pesos.detach()

        # Paso 4: Multiplicar por Values
        # (batch, num_cabezas, seq_len, seq_len) × (batch, num_cabezas, seq_len, d_cabeza)
        # → (batch, num_cabezas, seq_len, d_cabeza)
        contexto = torch.matmul(pesos, V)

        # Paso 5: Concatenar cabezas
        # (batch, num_cabezas, seq_len, d_cabeza) → (batch, seq_len, d_modelo)
        contexto = contexto.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_modelo
        )

        # Paso 6: Proyección de salida
        salida = self.W_o(contexto)

        return salida


class FeedForward(nn.Module):
    """
    Red Feed-Forward posicional.

    NOTA EDUCATIVA:
        Mientras la atención captura RELACIONES entre tokens,
        el Feed-Forward captura TRANSFORMACIONES de cada token
        individualmente.

        En el contexto de código C#:
        - La atención aprende: "public" seguido de "class" = declaración de clase
        - El FF aprende: "int" es un tipo numérico, "string" es texto

        Usamos GELU (Gaussian Error Linear Unit) como activación,
        que funciona mejor que ReLU para modelos de lenguaje.
    """

    def __init__(self, config: ConfiguracionTransformer):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(config.d_modelo, config.dim_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_ff, config.d_modelo),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)


class CapaTransformer(nn.Module):
    """
    Una capa completa del Transformer (Attention + FF + Residual + Norm).

    NOTA EDUCATIVA — CONEXIONES RESIDUALES:
        Las conexiones residuales (x + capa(x)) son CRUCIALES porque:
        1. Permiten que el gradiente fluya sin degradarse (entrenamiento estable)
        2. Cada capa puede REFINAR la representación sin perder información
        3. Las capas inferiores preservan información "cruda" del embedding

        Para código C#, esto significa que la capa 4 (última) todavía
        puede "ver" la información original del embedding (keywords, tipos)
        a través de las conexiones residuales.

    PRE-NORM vs POST-NORM:
        Usamos Pre-LayerNorm (normalizar ANTES de la subcapa),
        que es más estable para entrenar que Post-LayerNorm.
        GPT-2 y modelos modernos usan Pre-Norm.
    """

    def __init__(self, config: ConfiguracionTransformer):
        super().__init__()

        self.atencion = AtencionMultiCabeza(config)
        self.ff = FeedForward(config)

        # LayerNorm antes de cada subcapa (Pre-Norm)
        self.norm1 = nn.LayerNorm(config.d_modelo)
        self.norm2 = nn.LayerNorm(config.d_modelo)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mascara: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FLUJO:
            x → norm1 → atención → + residual → norm2 → ff → + residual → salida
        """
        # Sub-capa 1: Self-Attention con conexión residual
        residual = x
        x = self.norm1(x)
        x = self.atencion(x, mascara)
        x = self.dropout(x)
        x = residual + x

        # Sub-capa 2: Feed-Forward con conexión residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


# ============================================================================
# MODELO PRINCIPAL
# ============================================================================

class TransformerCodeGen(nn.Module):
    """
    Transformer completo para generación de código C#.

    ARQUITECTURA FINAL:
        ┌──────────────────────────────────────────┐
        │        Input: IDs de tokens C#            │
        │  [public, class, Product, {, NL, ...]     │
        ├──────────────────────────────────────────┤
        │  Embedding de Tokens + Posicional         │
        │  Cada token → vector de 256 dimensiones   │
        ├──────────────────────────────────────────┤
        │  Capa 1: Atención + FF                    │
        │  Aprende: relaciones locales (tipo-var)   │
        ├──────────────────────────────────────────┤
        │  Capa 2: Atención + FF                    │
        │  Aprende: relaciones de bloque ({...})    │
        ├──────────────────────────────────────────┤
        │  Capa 3: Atención + FF                    │
        │  Aprende: relaciones de método/clase      │
        ├──────────────────────────────────────────┤
        │  Capa 4: Atención + FF                    │
        │  Aprende: patrones arquitectónicos        │
        ├──────────────────────────────────────────┤
        │  Head de lenguaje: d_modelo → tam_vocab   │
        │  → Probabilidad de cada posible token     │
        │  → Seleccionar el más probable            │
        ├──────────────────────────────────────────┤
        │  Output: siguiente token predicho         │
        │  Ejemplo: dado "public class " → "Product"│
        └──────────────────────────────────────────┘

    COMPARACIÓN CON MODELOS DE TEXTO NATURAL:
        Un modelo de texto natural con la misma arquitectura generaría
        prosa, no código. La diferencia está en:
        1. El vocabulario (tokens de código vs palabras)
        2. Los datos de entrenamiento (código vs texto)
        3. El soporte FIM (no existe en modelos de texto estándar)
        4. Los patrones de atención aprendidos (estructura de código)
    """

    def __init__(self, config: ConfiguracionTransformer):
        super().__init__()

        self.config = config

        # 1. Embeddings
        self.embeddings = EmbeddingConPosicion(config)

        # 2. Capas del Transformer (4 capas)
        self.capas = nn.ModuleList([
            CapaTransformer(config) for _ in range(config.num_capas)
        ])

        # 3. Normalización final
        self.norm_final = nn.LayerNorm(config.d_modelo)

        # 4. Head de lenguaje (predice el siguiente token)
        # Mapea de d_modelo a tam_vocabulario
        self.head_lm = nn.Linear(config.d_modelo, config.tam_vocabulario, bias=False)

        # Weight tying: compartir pesos entre embedding y head
        # Esto reduce parámetros y mejora el rendimiento
        # La intuición: si el embedding de "class" es similar al de "struct",
        # entonces la probabilidad de generar "struct" cuando el contexto
        # espera "class" debería ser alta (son intercambiables en muchos casos)
        self.head_lm.weight = self.embeddings.embedding_token.weight

        # Inicializar pesos
        self.apply(self._inicializar_pesos)

        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🤖 Modelo TransformerCodeGen creado:")
        print(f"   Capas: {config.num_capas}")
        print(f"   d_modelo: {config.d_modelo}")
        print(f"   Cabezas: {config.num_cabezas}")
        print(f"   Vocabulario: {config.tam_vocabulario}")
        print(f"   Parámetros totales: {total_params:,}")

    def _inicializar_pesos(self, modulo):
        """
        Inicializa pesos del modelo.

        NOTA EDUCATIVA:
            La inicialización de pesos es CRUCIAL para el entrenamiento.
            - Pesos muy grandes → gradientes explotan
            - Pesos muy pequeños → gradientes desaparecen

            Usamos inicialización normal con std=0.02, que es el
            estándar para modelos Transformer (propuesto por GPT-2).
        """
        if isinstance(modulo, nn.Linear):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                torch.nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, nn.Embedding):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mascara: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modelo.

        Args:
            token_ids: (batch, seq_len) IDs de tokens de entrada
            labels: (batch, seq_len) IDs de tokens objetivo (para entrenamiento)
            mascara: Máscara de atención opcional

        Returns:
            Dict con:
            - "logits": (batch, seq_len, vocab_size) puntajes sin normalizar
            - "loss": escalar, solo si labels es proporcionado
            - "hidden_states": (batch, seq_len, d_modelo) estados ocultos

        NOTA EDUCATIVA — LOGITS Y LOSS:
            Los logits son puntajes sin normalizar para cada token del vocabulario.
            Para obtener probabilidades, se aplica softmax:
                P(siguiente_token) = softmax(logits)

            El loss es la CROSS-ENTROPY entre las probabilidades predichas
            y el token real. El modelo aprende minimizando este loss.

            Ejemplo:
                Input:  "public class "
                Logits: [0.1, 0.3, ..., 2.5, ...] (un número por token)
                Label:  token_id de "Product"
                Loss:   -log(P("Product")) = penalización por no predecir "Product"
        """
        # 1. Obtener embeddings
        x = self.embeddings(token_ids)

        # 2. Pasar por las 4 capas del Transformer
        for capa in self.capas:
            x = capa(x, mascara)

        # 3. Normalización final
        estados_ocultos = self.norm_final(x)

        # 4. Proyectar a vocabulario (logits)
        logits = self.head_lm(estados_ocultos)

        resultado = {
            "logits": logits,
            "hidden_states": estados_ocultos,
        }

        # 5. Calcular loss si hay labels
        if labels is not None:
            # Shift: predecimos el token SIGUIENTE
            # Input:  [BOS, public, class, Product]
            # Labels: [public, class, Product, EOS]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.tam_vocabulario),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id  # No penalizar padding
            )
            resultado["loss"] = loss

        return resultado

    # ================================================================
    # GENERACIÓN DE CÓDIGO
    # ================================================================

    @torch.no_grad()
    def generar(
        self,
        token_ids_inicio: torch.Tensor,
        max_tokens: int = 200,
        temperatura: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        tokens_parada: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Genera código C# token por token.

        NOTA EDUCATIVA — ESTRATEGIAS DE MUESTREO:

        TEMPERATURA:
            Controla la "creatividad" del modelo.
            - T=0.0: siempre elige el token más probable (determinístico)
            - T=0.8: algo de variedad (bueno para código)
            - T=1.5: muy variado (puede generar código incoherente)

            Para CÓDIGO, temperaturas bajas (0.2-0.8) funcionan mejor
            porque el código tiene menos opciones válidas que el texto.
            "public class" SIEMPRE debe ser seguido por un nombre válido.

        TOP-K:
            Solo considerar los K tokens más probables.
            Esto evita que el modelo genere tokens muy improbables.
            K=50 es un buen balance para código.

        TOP-P (Nucleus Sampling):
            Solo considerar tokens cuya probabilidad acumulada ≤ p.
            Adapta K dinámicamente según la distribución.
            p=0.9 significa: considerar tokens que cubren 90% de la masa
            de probabilidad.

        tokens_parada:
            IDs de tokens que detienen la generación.
            Típicamente: EOS, "}" (fin de clase/método).

        Args:
            token_ids_inicio: Tensor con la secuencia semilla
            max_tokens: Máximo de tokens a generar
            temperatura: Control de aleatoriedad (0.0 = determinístico)
            top_k: Número de tokens candidatos
            top_p: Umbral de probabilidad acumulada
            tokens_parada: Lista de token IDs que detienen la generación

        Returns:
            (tokens_generados, info_por_paso)
            info_por_paso contiene datos para visualización
        """
        self.eval()
        generados = token_ids_inicio.clone()
        info_pasos = []

        for paso in range(max_tokens):
            # Truncar si excede max_longitud
            entrada = generados[:, -self.config.max_longitud:]

            # Forward pass
            resultado = self.forward(entrada)
            logits = resultado["logits"][:, -1, :]  # Solo el último token

            # Aplicar temperatura
            if temperatura > 0:
                logits = logits / temperatura

            # Top-K filtering
            if top_k > 0:
                valores, indices = torch.topk(logits, top_k)
                logits[logits < valores[:, -1:]] = float('-inf')

            # Top-P filtering (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs_acumuladas = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Eliminar tokens con probabilidad acumulada > top_p
                indices_eliminar = probs_acumuladas > top_p
                indices_eliminar[:, 1:] = indices_eliminar[:, :-1].clone()
                indices_eliminar[:, 0] = False

                indices_originales = sorted_indices[indices_eliminar]
                logits[:, indices_originales] = float('-inf')

            # Muestrear token
            probabilidades = F.softmax(logits, dim=-1)
            siguiente_token = torch.multinomial(probabilidades, 1)

            # Guardar información del paso para visualización
            top_probs, top_ids = torch.topk(probabilidades, min(10, probabilidades.size(-1)))
            info_pasos.append({
                "paso": paso,
                "token_id": siguiente_token.item(),
                "probabilidad": probabilidades[0, siguiente_token.item()].item(),
                "top_candidatos": [
                    {"id": tid.item(), "prob": tp.item()}
                    for tid, tp in zip(top_ids[0], top_probs[0])
                ],
            })

            # Agregar token generado
            generados = torch.cat([generados, siguiente_token], dim=1)

            # Verificar tokens de parada
            if tokens_parada and siguiente_token.item() in tokens_parada:
                break

        return generados, info_pasos

    def generar_con_contexto_rag(
        self,
        prompt_ids: torch.Tensor,
        contexto_ids: torch.Tensor,
        max_tokens: int = 200,
        temperatura: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Genera código usando contexto recuperado por RAG.

        NOTA EDUCATIVA — RAG + GENERACIÓN:
            El contexto recuperado se PREPONE al prompt del usuario.
            Esto permite que el modelo "vea" ejemplos relevantes
            antes de generar nuevo código.

            Formato:
            [BOS] [contexto_rag] [SEPARATOR] [prompt_usuario] → generación

            ¿Por qué funciona?
            Porque el mecanismo de atención del Transformer puede
            "mirar" los tokens del contexto RAG y usarlos como
            referencia para generar código coherente.

            Es como cuando un desarrollador abre un archivo existente
            del proyecto antes de escribir código nuevo.
        """
        # Concatenar contexto + prompt
        entrada = torch.cat([contexto_ids, prompt_ids], dim=1)

        # Truncar si es necesario
        if entrada.size(1) > self.config.max_longitud - max_tokens:
            # Priorizar el prompt sobre el contexto
            max_contexto = self.config.max_longitud - max_tokens - prompt_ids.size(1)
            contexto_truncado = contexto_ids[:, -max_contexto:]
            entrada = torch.cat([contexto_truncado, prompt_ids], dim=1)

        return self.generar(
            entrada,
            max_tokens=max_tokens,
            temperatura=temperatura,
            top_k=top_k,
            top_p=top_p,
        )

    # ================================================================
    # UTILIDADES PARA VISUALIZACIÓN
    # ================================================================

    def obtener_pesos_atencion(self) -> List[Optional[torch.Tensor]]:
        """
        Retorna los pesos de atención de todas las capas.

        NOTA EDUCATIVA:
            Los pesos de atención muestran A QUÉ tokens presta
            atención el modelo en cada posición. Visualizarlos
            nos permite entender QUÉ aprende el modelo:

            - ¿Presta atención a las keywords cercanas?
            - ¿Presta atención a la llave de apertura?
            - ¿Presta atención al tipo declarado?

            En modelos de código, típicamente vemos:
            - Atención fuerte a "{" cuando genera "}"
            - Atención fuerte al tipo en asignaciones
            - Atención fuerte a "async" cuando genera "await"
        """
        pesos = []
        for capa in self.capas:
            pesos.append(capa.atencion.pesos_atencion)
        return pesos

    def contar_parametros(self) -> Dict[str, int]:
        """Cuenta parámetros por componente."""
        conteo = {}
        for nombre, param in self.named_parameters():
            componente = nombre.split('.')[0]
            conteo[componente] = conteo.get(componente, 0) + param.numel()

        conteo["total"] = sum(p.numel() for p in self.parameters())
        conteo["entrenables"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return conteo
