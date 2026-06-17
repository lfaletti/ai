# ============================================================================
# datos/__init__.py
# Módulo de datos: contiene el dataset sintético de C# y la base de
# conocimiento para RAG.
# ============================================================================
from datos.dataset_csharp import DatasetCSharp, generar_dataset_completo
from datos.base_conocimiento import BaseConocimientoRAG, construir_base_conocimiento
