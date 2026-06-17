"""
============================================================================
examples/ejemplos_uso.py — Ejemplos Prácticos del Generador de Código C#
============================================================================

Este archivo contiene ejemplos funcionales que demuestran cada
capacidad del sistema. Puede ejecutarse independientemente después
de haber corrido main.py al menos una vez.

EJEMPLOS INCLUIDOS:
    1. Autocompletado de clases C#
    2. Generación desde comentario XML
    3. Fill-in-the-Middle (FIM)
    4. RAG: búsqueda de contexto + generación
    5. Comparación con/sin RAG
    6. Análisis de tokenización
============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datos.dataset_csharp import generar_dataset_completo
from datos.base_conocimiento import construir_base_conocimiento
from modelo.tokenizador_csharp import TokenizadorCSharp
from rag.sistema_rag import SistemaRAG
from rag.embeddings import GeneradorEmbeddings


def ejemplo_1_tokenizacion():
    """
    EJEMPLO 1: Análisis detallado de tokenización C#.

    Muestra cómo el tokenizador divide código C# en tokens
    semánticamente significativos, diferente a un tokenizador genérico.
    """
    print("\n" + "=" * 60)
    print("📝 EJEMPLO 1: Tokenización Especializada para C#")
    print("=" * 60)

    tokenizador = TokenizadorCSharp(tam_vocab_max=4000)

    # Construir vocabulario con datos sintéticos
    dataset = generar_dataset_completo(tamano=50, semilla=42)
    tokenizador.construir_vocabulario(dataset.obtener_textos_entrenamiento())

    # Ejemplos de tokenización
    ejemplos = [
        # Ejemplo 1: Método async con genéricos
        "public async Task<List<ProductDto>> GetAllProductsAsync(int page = 1)",

        # Ejemplo 2: LINQ con lambda
        "var activos = items.Where(x => x.IsActive).OrderBy(x => x.Name).ToList();",

        # Ejemplo 3: Atributos de ASP.NET Core
        '[HttpGet("{id}")]\npublic async Task<ActionResult<ProductDto>> GetById(int id)',

        # Ejemplo 4: Null-conditional y null-coalescing
        "var nombre = usuario?.Perfil?.NombreCompleto ?? \"Sin nombre\";",
    ]

    for i, codigo in enumerate(ejemplos, 1):
        stats = tokenizador.obtener_estadisticas_tokenizacion(codigo)
        tokens = stats["tokens"]

        print(f"\n--- Ejemplo {i} ---")
        print(f"   Código: {codigo}")
        print(f"   Tokens ({stats['total_tokens']}): {tokens[:15]}...")
        print(f"   Únicos: {stats['tokens_unicos']}")
        print(f"   Ratio compresión: {stats['ratio_compresion']:.1f} chars/token")
        print(f"   Keywords C#: {stats['clasificacion']['keywords']}")

    # Demostrar PascalCase splitting
    print("\n--- PascalCase Splitting ---")
    idents = ["GetAllProductsAsync", "IOrderRepository", "CreateProductDto",
              "HandleHttpRequestException"]
    for ident in idents:
        partes = tokenizador._dividir_pascal_case(ident)
        print(f"   {ident} → {partes}")


def ejemplo_2_fim():
    """
    EJEMPLO 2: Fill-in-the-Middle.

    Demuestra cómo FIM permite completar código en medio de un archivo,
    no solo al final. Esta es la base de un copilot útil.
    """
    print("\n" + "=" * 60)
    print("📝 EJEMPLO 2: Fill-in-the-Middle (FIM)")
    print("=" * 60)

    tokenizador = TokenizadorCSharp(tam_vocab_max=4000)
    dataset = generar_dataset_completo(tamano=50, semilla=42)
    tokenizador.construir_vocabulario(dataset.obtener_textos_entrenamiento())

    # Código completo
    codigo = """public async Task<ProductDto> GetByIdAsync(int id)
{
    var product = await _repository.GetByIdAsync(id);
    if (product == null)
        throw new KeyNotFoundException($"Product {id} not found");
    return _mapper.Map<ProductDto>(product);
}"""

    resultado = tokenizador.preparar_fim(codigo)

    print(f"\n   Código original ({len(codigo)} chars):")
    print(f"   {codigo[:100]}...")
    print(f"\n   PREFIX: ...{resultado['prefijo'][-60:]}...")
    print(f"   MIDDLE: {resultado['middle']}")
    print(f"   SUFFIX: {resultado['sufijo'][:60]}...")
    print(f"\n   Posición cursor: {resultado['posicion_cursor']}")

    # Codificar FIM
    ids = tokenizador.codificar_fim(
        resultado['prefijo'],
        resultado['sufijo'],
        resultado['middle']
    )
    print(f"   Secuencia FIM: {len(ids)} tokens")
    print(f"   Primeros IDs: {ids[:10]}...")


def ejemplo_3_rag_busqueda():
    """
    EJEMPLO 3: Sistema RAG — Búsqueda de código relevante.

    Demuestra cómo el sistema RAG encuentra snippets de código
    relevantes para una query en lenguaje natural.
    """
    print("\n" + "=" * 60)
    print("📝 EJEMPLO 3: Sistema RAG — Búsqueda de Código")
    print("=" * 60)

    # Construir base de conocimiento
    base = construir_base_conocimiento()
    sistema = SistemaRAG(dimension_embeddings=256, top_k=3)
    sistema.indexar(base)

    # Queries de ejemplo
    queries = [
        "crear endpoint API REST para productos",
        "servicio de pedidos con cálculo de totales LINQ",
        "middleware de autenticación JWT",
        "repositorio genérico con Entity Framework",
        "configurar inyección de dependencias startup",
        "consultas para reportes con agrupación",
    ]

    for query in queries:
        resultado = sistema.buscar(query)
        print(f"\n   Query: '{query}'")
        print(f"   Tiempo: {resultado.tiempo_busqueda_ms:.2f}ms")
        for doc in resultado.documentos_recuperados:
            print(f"   → #{doc['rank']} [{doc['categoria']}] "
                  f"(sim: {doc['similitud']:.2f}) "
                  f"{doc['descripcion'][:50]}...")


def ejemplo_4_embeddings():
    """
    EJEMPLO 4: Análisis de embeddings.

    Muestra cómo los embeddings capturan la semántica del código,
    permitiendo encontrar código similar incluso con palabras diferentes.
    """
    print("\n" + "=" * 60)
    print("📝 EJEMPLO 4: Análisis de Embeddings")
    print("=" * 60)

    gen = GeneradorEmbeddings(dimension=256)

    textos = [
        "Controller REST para gestionar productos con CRUD",
        "Endpoint API HTTP para manejar items con operaciones básicas",
        "Conexión a base de datos SQL Server con Entity Framework",
        "Servicio de autenticación JWT con tokens de acceso",
        "Consulta LINQ con agrupación y estadísticas",
    ]

    gen.construir(textos)

    print("\n   Similitudes coseno entre pares de textos:")
    print("   (1.0 = idénticos, 0.0 = sin relación)\n")

    embeddings = gen.generar_embeddings_batch(textos)

    for i in range(len(textos)):
        for j in range(i + 1, len(textos)):
            sim = gen.similitud_coseno(embeddings[i], embeddings[j])
            label_i = textos[i][:40]
            label_j = textos[j][:40]
            barra = "█" * int(sim * 20)
            print(f"   [{label_i}...]\n"
                  f"   [{label_j}...]\n"
                  f"   Similitud: {sim:.3f} {barra}\n")


def ejemplo_5_dataset():
    """
    EJEMPLO 5: Exploración del dataset sintético.

    Muestra las estadísticas y la composición del dataset
    de código C# generado sintéticamente.
    """
    print("\n" + "=" * 60)
    print("📝 EJEMPLO 5: Dataset Sintético de C#")
    print("=" * 60)

    dataset = generar_dataset_completo(tamano=100, semilla=42)
    stats = dataset.obtener_estadisticas()

    print(f"\n   📊 Estadísticas del dataset:")
    print(f"   Total snippets: {stats['total_snippets']}")
    print(f"   Total líneas: {stats['total_lineas']}")
    print(f"   Total caracteres: {stats['total_caracteres']}")
    print(f"   Complejidad promedio: {stats['complejidad_promedio']:.1f}/5")
    print(f"   Con LINQ: {stats['con_linq']}")
    print(f"   Con async: {stats['con_async']}")
    print(f"   Con generics: {stats['con_generics']}")
    print(f"\n   Por categoría:")
    for cat, count in sorted(stats['por_categoria'].items()):
        barra = "█" * count
        print(f"      {cat:15s}: {count:3d} {barra}")

    # Mostrar un snippet de ejemplo
    print(f"\n   📝 Ejemplo de snippet (primero del dataset):")
    snippet = dataset.snippets[0]
    print(f"   Categoría: {snippet.categoria}")
    print(f"   Patrón: {snippet.patron}")
    print(f"   Complejidad: {snippet.complejidad}/5")
    print(f"   Keywords: {snippet.palabras_clave}")
    print(f"   Código (primeras 5 líneas):")
    for linea in snippet.codigo.split('\n')[:5]:
        print(f"      {linea}")


if __name__ == "__main__":
    print("🔧 Ejemplos del Generador de Código C#")
    print("=" * 60)

    ejemplo_1_tokenizacion()
    ejemplo_2_fim()
    ejemplo_3_rag_busqueda()
    ejemplo_4_embeddings()
    ejemplo_5_dataset()

    print("\n\n✅ Todos los ejemplos ejecutados exitosamente.")
    print("Para ver el pipeline completo con entrenamiento, ejecute: python main.py")
