# ============================================================================
# modelo/__init__.py
# Módulo del modelo: contiene el tokenizador de C#, el Transformer
# y la pipeline de entrenamiento/inferencia.
# ============================================================================
from modelo.tokenizador_csharp import TokenizadorCSharp
from modelo.transformer import TransformerCodeGen, ConfiguracionTransformer
from modelo.entrenamiento import EntrenadorModelo
