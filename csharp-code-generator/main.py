"""
============================================================================
main.py — Pipeline Principal del Generador de Código C#
============================================================================

PROPÓSITO:
    Este es el punto de entrada principal que orquesta todo el sistema:
    1. Genera el dataset sintético de C#
    2. Construye y entrena el tokenizador
    3. Entrena el modelo Transformer
    4. Configura el sistema RAG con FAISS
    5. Ejecuta ejemplos de uso
    6. Exporta datos para la visualización HTML/JS

USO:
    python main.py              # Ejecutar pipeline completo
    python main.py --solo-viz   # Solo regenerar visualización
    python main.py --rapido     # Entrenamiento rápido (2 epochs)
============================================================================
"""

import sys
import os
import json
import torch
import argparse
import random
import numpy as np

# Agregar directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datos.dataset_csharp import DatasetCSharp, generar_dataset_completo
from datos.base_conocimiento import BaseConocimientoRAG, construir_base_conocimiento
from modelo.tokenizador_csharp import TokenizadorCSharp
from modelo.transformer import TransformerCodeGen, ConfiguracionTransformer
from modelo.entrenamiento import EntrenadorModelo, ConfigEntrenamiento
from rag.sistema_rag import SistemaRAG
from rag.embeddings import GeneradorEmbeddings
from utils.exportar_visualizacion import ExportadorVisualizacion


def fijar_semillas(semilla: int = 42):
    """Fija semillas para reproducibilidad total."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def paso_1_generar_dataset() -> DatasetCSharp:
    """
    PASO 1: Generar dataset sintético de código C#.

    Genera ~100 snippets de código C# profesional cubriendo:
    - DTOs, Interfaces, Services, Controllers
    - LINQ avanzado, async/await
    - Middleware, Validadores, Event Handlers
    """
    print("\n" + "=" * 60)
    print("📦 PASO 1: Generando Dataset Sintético de C#")
    print("=" * 60)

    dataset = generar_dataset_completo(tamano=100, semilla=42)
    stats = dataset.obtener_estadisticas()

    print(f"\n📊 Estadísticas del dataset:")
    print(f"   Total snippets: {stats['total_snippets']}")
    print(f"   Total líneas de código: {stats['total_lineas']}")
    print(f"   Total caracteres: {stats['total_caracteres']}")
    print(f"   Complejidad promedio: {stats['complejidad_promedio']:.1f}/5")
    print(f"   Con LINQ: {stats['con_linq']}")
    print(f"   Con async: {stats['con_async']}")
    print(f"   Con generics: {stats['con_generics']}")
    print(f"   Por categoría:")
    for cat, count in sorted(stats['por_categoria'].items()):
        print(f"      {cat}: {count}")

    return dataset


def paso_2_construir_tokenizador(dataset: DatasetCSharp) -> TokenizadorCSharp:
    """
    PASO 2: Construir el tokenizador especializado para C#.

    Analiza el corpus, construye vocabulario, y demuestra la tokenización.
    """
    print("\n" + "=" * 60)
    print("🔤 PASO 2: Construyendo Tokenizador de C#")
    print("=" * 60)

    tokenizador = TokenizadorCSharp(tam_vocab_max=4000)
    textos = dataset.obtener_textos_entrenamiento()
    tokenizador.construir_vocabulario(textos)

    # Demostrar tokenización con un ejemplo
    ejemplo = "public async Task<List<ProductDto>> GetAllAsync(int page = 1)"
    tokens = tokenizador.tokenizar(ejemplo)
    print(f"\n📝 Ejemplo de tokenización:")
    print(f"   Input:  {ejemplo}")
    print(f"   Tokens: {tokens[:20]}...")
    print(f"   IDs:    {tokenizador.codificar(ejemplo)[:20]}...")

    # Guardar vocabulario
    os.makedirs("checkpoints", exist_ok=True)
    tokenizador.guardar("checkpoints/vocabulario.json")

    return tokenizador


def paso_3_entrenar_modelo(
    dataset: DatasetCSharp,
    tokenizador: TokenizadorCSharp,
    epochs: int = 5
) -> tuple:
    """
    PASO 3: Entrenar el modelo Transformer de 4 capas.
    """
    print("\n" + "=" * 60)
    print("🧠 PASO 3: Entrenando Modelo Transformer (4 capas)")
    print("=" * 60)

    # Configurar modelo
    config_modelo = ConfiguracionTransformer(
        tam_vocabulario=tokenizador.tam_vocabulario,
        d_modelo=256,
        num_cabezas=8,
        num_capas=4,
        dim_ff=1024,
        max_longitud=512,
        dropout=0.1,
        pad_token_id=tokenizador.pad_id,
        guardar_atencion=True,
    )

    modelo = TransformerCodeGen(config_modelo)

    # Configurar entrenamiento
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    config_entrenamiento = ConfigEntrenamiento(
        learning_rate=3e-4,
        batch_size=4,
        epochs=epochs,
        warmup_steps=50,
        max_longitud=256,  # Reducido para entrenamiento rápido
        proporcion_fim=0.3,
        dispositivo=dispositivo,
        log_cada=20,
    )

    entrenador = EntrenadorModelo(modelo, tokenizador, config_entrenamiento)

    # Entrenar
    textos = dataset.obtener_textos_entrenamiento()
    resultado = entrenador.entrenar(textos)

    return modelo, entrenador, resultado


def paso_4_configurar_rag() -> SistemaRAG:
    """
    PASO 4: Configurar el sistema RAG con FAISS.
    """
    print("\n" + "=" * 60)
    print("🔍 PASO 4: Configurando Sistema RAG con FAISS")
    print("=" * 60)

    # Construir base de conocimiento
    base = construir_base_conocimiento()
    stats = base.obtener_estadisticas()

    print(f"\n📚 Base de conocimiento:")
    print(f"   Documentos: {stats['total_documentos']}")
    print(f"   Categorías: {list(stats['categorias'].keys())}")

    # Configurar y cargar sistema RAG
    sistema_rag = SistemaRAG(dimension_embeddings=256, top_k=3)
    sistema_rag.indexar(base)

    # Demostrar búsqueda
    queries_demo = [
        "crear endpoint API REST para productos",
        "servicio de pedidos con cálculo de totales",
        "configurar inyección de dependencias en startup",
    ]

    print("\n📝 Demostración de búsqueda RAG:")
    for query in queries_demo:
        resultado = sistema_rag.buscar(query)
        print(f"\n   Query: '{query}'")
        print(f"   Tiempo: {resultado.tiempo_busqueda_ms:.2f}ms")
        for doc in resultado.documentos_recuperados[:2]:
            print(f"   → [{doc['similitud']:.2f}] {doc['descripcion'][:60]}...")

    return sistema_rag


def paso_5_ejemplos_uso(entrenador, sistema_rag, tokenizador):
    """
    PASO 5: Ejecutar ejemplos de uso del sistema completo.
    """
    print("\n" + "=" * 60)
    print("🚀 PASO 5: Ejemplos de Uso")
    print("=" * 60)

    # ----- Ejemplo 1: Autocompletado de clase -----
    print("\n📝 Ejemplo 1: Autocompletado de clase")
    print("-" * 40)
    resultado = entrenador.completar_codigo(
        "public class ProductService : IProductService\n{",
        max_tokens=100,
        temperatura=0.7,
    )
    print(f"   Prompt: {resultado['prompt']}")
    print(f"   Generado ({resultado['tokens_generados']} tokens):")
    print(f"   {resultado['codigo_generado'][:300]}")

    # ----- Ejemplo 2: Generación desde comentario XML -----
    print("\n📝 Ejemplo 2: Generación desde comentario XML")
    print("-" * 40)
    resultado = entrenador.completar_codigo(
        "/// <summary>\n/// Obtiene un producto por su ID.\n/// </summary>\npublic async",
        max_tokens=80,
        temperatura=0.7,
    )
    print(f"   Generado: {resultado['codigo_generado'][:300]}")

    # ----- Ejemplo 3: Fill-in-the-Middle -----
    print("\n📝 Ejemplo 3: Fill-in-the-Middle (FIM)")
    print("-" * 40)
    resultado = entrenador.completar_fim(
        prefijo="public async Task<ProductDto> GetByIdAsync(int id)\n{\n    ",
        sufijo="\n    return _mapper.Map<ProductDto>(product);\n}",
        max_tokens=50,
        temperatura=0.7,
    )
    print(f"   Prefijo: {resultado['prefijo']}")
    print(f"   Sufijo: {resultado['sufijo']}")
    print(f"   Middle generado: {resultado['middle_generado'][:200]}")

    # ----- Ejemplo 4: RAG - Crear endpoint REST -----
    print("\n📝 Ejemplo 4: RAG - Crear endpoint API REST")
    print("-" * 40)
    comparacion = sistema_rag.comparar_con_sin_rag(
        query="crear endpoint API REST para gestionar pedidos",
        modelo=entrenador.modelo,
        tokenizador=tokenizador,
        max_tokens=100,
        temperatura=0.7,
    )

    return comparacion


def paso_6_exportar_visualizacion(
    tokenizador, modelo, entrenador, sistema_rag, historial, comparacion_rag
):
    """
    PASO 6: Exportar datos para la visualización HTML/JS.
    """
    print("\n" + "=" * 60)
    print("📊 PASO 6: Exportando Datos para Visualización")
    print("=" * 60)

    exportador = ExportadorVisualizacion("visualizacion")

    # 1. Datos de tokenización
    codigos_ejemplo = [
        "public async Task<List<ProductDto>> GetAllAsync(int page = 1)",
        "[HttpGet(\"{id}\")]\npublic async Task<ActionResult<ProductDto>> GetById(int id)",
        "var result = items.Where(x => x.IsActive).OrderBy(x => x.Name).ToList();",
        "public class OrderService : IOrderService\n{\n    private readonly IOrderRepository _repo;\n}",
    ]
    exportador.exportar_tokenizacion(tokenizador, codigos_ejemplo)

    # 2. Datos de atención
    codigos_atencion = [
        "public class Product { public int Id { get; set; } }",
        "var items = list.Where(x => x.IsActive).ToList();",
    ]
    exportador.exportar_atencion(modelo, tokenizador, codigos_atencion)

    # 3. Datos de generación token por token
    prompts_generacion = [
        "public class",
        "[HttpGet]",
        "public async Task<",
    ]
    exportador.exportar_generacion(entrenador, prompts_generacion)

    # 4. Datos RAG
    queries_rag = [
        "crear endpoint API REST para productos",
        "servicio con lógica de negocio y validación",
        "consultas LINQ para reportes con agrupación",
        "middleware de autenticación JWT",
    ]
    exportador.exportar_rag(sistema_rag, queries_rag)

    # 5. Comparación con/sin RAG
    if comparacion_rag:
        exportador.exportar_comparacion_rag([comparacion_rag])

    # 6. Historial de entrenamiento
    if historial:
        exportador.exportar_historial_entrenamiento(historial)

    print("\n✅ Todos los datos exportados a ./visualizacion/")


def main():
    """Pipeline principal."""
    parser = argparse.ArgumentParser(description="Generador de Código C# Educativo")
    parser.add_argument("--rapido", action="store_true",
                       help="Entrenamiento rápido (2 epochs)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Número de epochs de entrenamiento")
    args = parser.parse_args()

    fijar_semillas(42)

    epochs = 2 if args.rapido else args.epochs

    # Ejecutar pipeline completo
    dataset = paso_1_generar_dataset()
    tokenizador = paso_2_construir_tokenizador(dataset)
    modelo, entrenador, resultado_entrenamiento = paso_3_entrenar_modelo(
        dataset, tokenizador, epochs=epochs
    )
    sistema_rag = paso_4_configurar_rag()
    comparacion_rag = paso_5_ejemplos_uso(entrenador, sistema_rag, tokenizador)
    paso_6_exportar_visualizacion(
        tokenizador, modelo, entrenador, sistema_rag,
        resultado_entrenamiento.get("historial", []),
        comparacion_rag,
    )

    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETO")
    print("=" * 60)
    print("Para ver la visualización interactiva:")
    print("  1. Abra visualizacion/index.html en un navegador")
    print("  2. O ejecute: python -m http.server 8080 -d visualizacion")
    print("=" * 60)


if __name__ == "__main__":
    main()
