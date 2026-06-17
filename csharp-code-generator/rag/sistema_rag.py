"""
============================================================================
rag/sistema_rag.py — Sistema RAG Completo con FAISS
============================================================================

PROPÓSITO EDUCATIVO:
    Implementación completa de Retrieval-Augmented Generation (RAG)
    para código C#. Este sistema:

    1. INDEXA snippets de código C# en una base vectorial FAISS
    2. BUSCA los snippets más relevantes para una query
    3. COMBINA los snippets como contexto para el modelo generador
    4. GENERA código informado por el contexto recuperado

¿QUÉ ES RAG?
    RAG = Retrieval-Augmented Generation
    Es una técnica que MEJORA los modelos generativos dándoles
    acceso a una base de conocimiento externa.

    SIN RAG: El modelo genera "de memoria" → puede alucinar
    CON RAG: El modelo consulta una base de datos → genera código basado en ejemplos reales

    Analogía: Es como la diferencia entre un programador que escribe
    de memoria vs uno que revisa la documentación antes de codificar.

¿POR QUÉ RAG ES ESPECIALMENTE BUENO PARA CÓDIGO?
    1. CONSISTENCIA: El código generado sigue los patrones del proyecto
    2. APIs CORRECTAS: No inventa métodos o clases que no existen
    3. CONVENCIONES: Mantiene el mismo estilo de código
    4. ACTUALIZACIONES: Si cambian las dependencias, solo hay que
       actualizar la base de conocimiento (no reentrenar el modelo)

¿QUÉ ES FAISS?
    FAISS (Facebook AI Similarity Search) es una librería de Facebook
    para búsqueda eficiente de vectores similares.

    Sin FAISS: buscar entre 1M de vectores toma O(N) → 1M comparaciones
    Con FAISS: usa índices optimizados → ~O(log N) → mucho más rápido

    Para nuestro prototipo educativo con ~100 snippets, la diferencia
    es mínima, pero en producción con millones de snippets, FAISS
    es ESENCIAL.
============================================================================
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from rag.embeddings import GeneradorEmbeddings
from datos.base_conocimiento import BaseConocimientoRAG, DocumentoRAG


@dataclass
class ResultadoRAG:
    """
    Resultado de una búsqueda RAG con información para visualización.

    NOTA EDUCATIVA:
        Almacenamos no solo el código recuperado, sino también:
        - La similitud (para visualizar un heat map)
        - La query original (para comparar con los resultados)
        - Métricas de tiempo (para mostrar la eficiencia de FAISS)
    """
    query: str                                    # Query original del usuario
    documentos_recuperados: List[Dict] = field(default_factory=list)
    embeddings_query: Optional[np.ndarray] = None # Para visualización
    tiempo_busqueda_ms: float = 0.0               # Tiempo de búsqueda
    contexto_generado: str = ""                   # Contexto combinado para el modelo


class SistemaRAG:
    """
    Sistema RAG completo para generación de código C#.

    FLUJO COMPLETO:
    ┌────────────────────────────────────────────────────────┐
    │  1. INDEXACIÓN (offline, una sola vez)                  │
    │     Snippets C# → Embeddings → Índice FAISS            │
    │                                                         │
    │  2. BÚSQUEDA (online, por cada query)                  │
    │     Query usuario → Embedding → FAISS → Top-K snippets │
    │                                                         │
    │  3. GENERACIÓN (online, después de búsqueda)           │
    │     Top-K snippets + Query → Modelo → Código generado   │
    └────────────────────────────────────────────────────────┘

    COMPARACIÓN CON/SIN RAG:
    ┌──────────────────────────────────────────────────────────────┐
    │  SIN RAG:                                                    │
    │  "crear endpoint REST" → Modelo → genera de memoria          │
    │  (puede usar convenciones incorrectas)                       │
    │                                                               │
    │  CON RAG:                                                     │
    │  "crear endpoint REST" → FAISS → encuentra ProductsController │
    │  → Modelo ve el ejemplo → genera código consistente           │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        dimension_embeddings: int = 256,
        top_k: int = 3,
    ):
        """
        Inicializa el sistema RAG.

        Args:
            dimension_embeddings: Dimensión de los vectores de embeddings.
                                  Debe coincidir con el generador de embeddings.
            top_k: Número de documentos a recuperar por defecto.

        NOTA EDUCATIVA — TOP-K:
            top_k controla cuántos snippets se recuperan:
            - k=1: solo el más relevante (riesgo de perder contexto)
            - k=3: buen balance entre relevancia y diversidad
            - k=10: mucho contexto pero puede incluir snippets irrelevantes

            Para código C#, k=3 funciona bien porque típicamente
            necesitamos ver 2-3 archivos relacionados (Controller,
            Service, DTO) para generar código coherente.
        """
        self.dimension = dimension_embeddings
        self.top_k = top_k

        # Componentes del sistema
        self.generador_embeddings = GeneradorEmbeddings(dimension=dimension_embeddings)
        self.base_conocimiento: Optional[BaseConocimientoRAG] = None

        # Índice FAISS (se crea al indexar)
        self.indice_faiss = None

        # Embeddings almacenados (para visualización)
        self.embeddings_almacenados: Optional[np.ndarray] = None

        # Flag de estado
        self._indexado = False

    def indexar(self, base_conocimiento: BaseConocimientoRAG):
        """
        Indexa la base de conocimiento en FAISS.

        PROCESO:
        1. Obtener textos de todos los documentos
        2. Construir modelo de embeddings (calcular IDF)
        3. Generar embeddings para cada documento
        4. Crear índice FAISS y agregar embeddings

        NOTA EDUCATIVA — TIPOS DE ÍNDICE FAISS:
            FAISS ofrece varios tipos de índices:

            - IndexFlatL2: búsqueda exacta por distancia L2
              ✅ Resultado exacto, ❌ lento con muchos documentos

            - IndexFlatIP: búsqueda exacta por producto interno
              ✅ Ideal para similitud coseno (con vectores normalizados)

            - IndexIVFFlat: índice con particiones (clusters)
              ✅ Más rápido, ❌ resultado aproximado

            - IndexHNSW: grafo de vecinos más cercanos
              ✅ Muy rápido, ❌ más memoria

            Usamos IndexFlatIP porque:
            1. Nuestro dataset es pequeño (~100 docs) → no necesitamos aproximación
            2. Nuestros embeddings están normalizados → IP = similitud coseno
            3. Es el más simple de entender educativamente
        """
        print("🔍 Indexando base de conocimiento en FAISS...")
        self.base_conocimiento = base_conocimiento

        # Paso 1: Obtener textos para embeddings
        textos = base_conocimiento.obtener_textos()

        if not textos:
            print("⚠️ Base de conocimiento vacía, no hay nada que indexar")
            return

        # Paso 2: Construir modelo de embeddings
        self.generador_embeddings.construir(textos)

        # Paso 3: Generar embeddings
        self.embeddings_almacenados = self.generador_embeddings.generar_embeddings_batch(textos)

        # Paso 4: Crear índice FAISS
        try:
            import faiss

            # IndexFlatIP = Inner Product (similar a coseno con vectores normalizados)
            self.indice_faiss = faiss.IndexFlatIP(self.dimension)

            # Normalizar embeddings para que IP = coseno
            faiss.normalize_L2(self.embeddings_almacenados)

            # Agregar embeddings al índice
            self.indice_faiss.add(self.embeddings_almacenados)

            self._indexado = True
            print(f"✅ Índice FAISS creado con {self.indice_faiss.ntotal} vectores")

        except ImportError:
            print("⚠️ FAISS no disponible, usando búsqueda por fuerza bruta")
            self._indexado = True  # Usaremos búsqueda numpy directa

        stats = base_conocimiento.obtener_estadisticas()
        print(f"   Documentos indexados: {stats['total_documentos']}")
        print(f"   Categorías: {list(stats['categorias'].keys())}")

    def buscar(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> ResultadoRAG:
        """
        Busca los snippets más relevantes para una query.

        FLUJO:
        1. Convertir query a embedding
        2. Buscar los top-K vectores más cercanos en FAISS
        3. Recuperar los documentos correspondientes
        4. Combinar como contexto para el modelo

        Args:
            query: Texto de búsqueda (ej: "crear endpoint REST para productos")
            top_k: Número de resultados a retornar

        Returns:
            ResultadoRAG con documentos recuperados y contexto generado
        """
        if not self._indexado:
            print("⚠️ El índice no ha sido construido. Llame a indexar() primero.")
            return ResultadoRAG(query=query)

        k = top_k or self.top_k
        import time
        inicio = time.time()

        # Paso 1: Generar embedding de la query
        embedding_query = self.generador_embeddings.generar_embedding(query)
        embedding_query = embedding_query.reshape(1, -1).astype(np.float32)

        # Normalizar para similitud coseno
        norma = np.linalg.norm(embedding_query)
        if norma > 0:
            embedding_query = embedding_query / norma

        # Paso 2: Buscar en FAISS (o fuerza bruta si FAISS no está disponible)
        if self.indice_faiss is not None:
            distancias, indices = self.indice_faiss.search(embedding_query, k)
            distancias = distancias[0]
            indices = indices[0]
        else:
            # Búsqueda por fuerza bruta con numpy
            similitudes = np.dot(self.embeddings_almacenados, embedding_query.T).flatten()
            indices = np.argsort(similitudes)[::-1][:k]
            distancias = similitudes[indices]

        tiempo_ms = (time.time() - inicio) * 1000

        # Paso 3: Recuperar documentos
        documentos = self.base_conocimiento.obtener_todos()
        docs_recuperados = []

        for i, (idx, dist) in enumerate(zip(indices, distancias)):
            if idx < len(documentos):
                doc = documentos[idx]
                docs_recuperados.append({
                    "rank": i + 1,
                    "id": doc.id,
                    "similitud": float(dist),
                    "categoria": doc.categoria,
                    "descripcion": doc.descripcion_natural,
                    "codigo": doc.contenido,
                    "etiquetas": doc.etiquetas,
                    "entidad": doc.entidad,
                    "patron": doc.patron,
                })

        # Paso 4: Generar contexto combinado
        contexto = self._generar_contexto(docs_recuperados)

        resultado = ResultadoRAG(
            query=query,
            documentos_recuperados=docs_recuperados,
            embeddings_query=embedding_query.flatten(),
            tiempo_busqueda_ms=tiempo_ms,
            contexto_generado=contexto,
        )

        return resultado

    def _generar_contexto(self, documentos: List[Dict]) -> str:
        """
        Combina los documentos recuperados en un contexto para el modelo.

        NOTA EDUCATIVA — FORMATO DEL CONTEXTO:
            El contexto se prepone al prompt del usuario para que
            el modelo lo "vea" a través de su mecanismo de atención.

            Formato:
            // [RAG] Ejemplo 1 (similitud: 0.85): Controller REST
            [código del ejemplo 1]
            // [RAG] Ejemplo 2 (similitud: 0.72): Service con LINQ
            [código del ejemplo 2]
            // [FIN RAG] Generar a partir de aquí:

            El modelo aprende a usar estos ejemplos como referencia.
            En producción, se usarían marcadores especiales en el
            entrenamiento para que el modelo distinga contexto RAG
            de código a generar.
        """
        partes = ["// === CONTEXTO RAG (snippets relevantes del proyecto) ===\n"]

        for doc in documentos:
            partes.append(
                f"// --- Ejemplo (similitud: {doc['similitud']:.2f}, "
                f"patrón: {doc['patron']}) ---\n"
            )
            # Limitar longitud del código para no exceder contexto del modelo
            codigo = doc['codigo']
            if len(codigo) > 500:
                codigo = codigo[:500] + "\n// ... (truncado)"
            partes.append(codigo + "\n\n")

        partes.append("// === FIN CONTEXTO RAG. Generar código a continuación: ===\n")

        return "\n".join(partes)

    # ================================================================
    # COMPARACIÓN CON/SIN RAG
    # ================================================================

    def comparar_con_sin_rag(
        self,
        query: str,
        modelo,
        tokenizador,
        max_tokens: int = 200,
        temperatura: float = 0.7,
    ) -> Dict:
        """
        Genera código con y sin RAG para comparar los resultados.

        NOTA EDUCATIVA:
            Esta función es CLAVE para demostrar el valor de RAG.
            Genera el mismo código de dos formas:

            1. SIN RAG: el modelo genera solo con el prompt
            2. CON RAG: el modelo genera con contexto recuperado

            Luego comparamos:
            - ¿El código con RAG es más coherente?
            - ¿Usa las mismas convenciones del proyecto?
            - ¿Los nombres de clases/métodos son consistentes?

        Args:
            query: Descripción de lo que se quiere generar
            modelo: Modelo TransformerCodeGen entrenado
            tokenizador: TokenizadorCSharp con vocabulario
            max_tokens: Tokens máximos por generación
            temperatura: Control de creatividad

        Returns:
            Dict con resultados de ambas generaciones y la comparación
        """
        import torch

        print(f"\n🔄 Comparando generación con/sin RAG para: '{query}'")
        print("=" * 60)

        # ----- Generación SIN RAG -----
        print("\n📝 Generación SIN RAG (solo modelo):")
        prompt_sin_rag = f"// {query}\n"
        ids_sin_rag = tokenizador.codificar(prompt_sin_rag, agregar_especiales=True)
        tensor_sin_rag = torch.tensor([ids_sin_rag])

        modelo.eval()
        with torch.no_grad():
            generados_sin_rag, info_sin_rag = modelo.generar(
                tensor_sin_rag,
                max_tokens=max_tokens,
                temperatura=temperatura,
            )

        codigo_sin_rag = tokenizador.decodificar(generados_sin_rag[0].tolist())
        print(f"   {codigo_sin_rag[:200]}...")

        # ----- Generación CON RAG -----
        print("\n📝 Generación CON RAG (modelo + contexto):")
        resultado_rag = self.buscar(query)

        print(f"   📊 Snippets recuperados: {len(resultado_rag.documentos_recuperados)}")
        for doc in resultado_rag.documentos_recuperados:
            print(f"      #{doc['rank']}: {doc['descripcion'][:60]}... "
                  f"(similitud: {doc['similitud']:.2f})")

        # Combinar contexto RAG + prompt
        prompt_con_rag = resultado_rag.contexto_generado + f"\n// {query}\n"
        ids_con_rag = tokenizador.codificar(prompt_con_rag, agregar_especiales=True)
        tensor_con_rag = torch.tensor([ids_con_rag])

        with torch.no_grad():
            generados_con_rag, info_con_rag = modelo.generar(
                tensor_con_rag,
                max_tokens=max_tokens,
                temperatura=temperatura,
            )

        codigo_con_rag = tokenizador.decodificar(generados_con_rag[0].tolist())
        print(f"   {codigo_con_rag[:200]}...")

        # ----- Comparación -----
        comparacion = {
            "query": query,
            "sin_rag": {
                "codigo": codigo_sin_rag,
                "tokens_generados": len(generados_sin_rag[0]) - len(ids_sin_rag),
                "info_pasos": info_sin_rag[:10],
            },
            "con_rag": {
                "codigo": codigo_con_rag,
                "tokens_generados": len(generados_con_rag[0]) - len(ids_con_rag),
                "info_pasos": info_con_rag[:10],
                "snippets_recuperados": resultado_rag.documentos_recuperados,
                "tiempo_busqueda_ms": resultado_rag.tiempo_busqueda_ms,
                "contexto": resultado_rag.contexto_generado,
            },
            "metricas": self._calcular_metricas_comparacion(
                codigo_sin_rag, codigo_con_rag, resultado_rag
            ),
        }

        print("\n📊 Métricas de comparación:")
        for k, v in comparacion["metricas"].items():
            print(f"   {k}: {v}")

        return comparacion

    def _calcular_metricas_comparacion(
        self, codigo_sin: str, codigo_con: str, resultado_rag: ResultadoRAG
    ) -> Dict:
        """
        Calcula métricas para comparar generación con/sin RAG.

        NOTA EDUCATIVA:
            Evaluar la calidad del código generado es MÁS DIFÍCIL que
            evaluar texto generado. No basta con medir perplexity.

            Métricas que usamos:
            1. Longitud del código (con RAG suele generar más)
            2. Presencia de patrones C# (keywords, atributos)
            3. Balance de llaves (coherencia sintáctica básica)
            4. Similitud con snippets recuperados (coherencia con proyecto)
        """
        keywords_csharp = {
            'public', 'private', 'class', 'async', 'await', 'Task',
            'return', 'var', 'new', 'void', 'string', 'int'
        }

        def contar_keywords(codigo):
            return sum(1 for kw in keywords_csharp if kw in codigo)

        def balance_llaves(codigo):
            return abs(codigo.count('{') - codigo.count('}'))

        return {
            "longitud_sin_rag": len(codigo_sin),
            "longitud_con_rag": len(codigo_con),
            "keywords_sin_rag": contar_keywords(codigo_sin),
            "keywords_con_rag": contar_keywords(codigo_con),
            "balance_llaves_sin_rag": balance_llaves(codigo_sin),
            "balance_llaves_con_rag": balance_llaves(codigo_con),
            "snippets_recuperados": len(resultado_rag.documentos_recuperados),
            "similitud_promedio": np.mean([
                d['similitud'] for d in resultado_rag.documentos_recuperados
            ]) if resultado_rag.documentos_recuperados else 0,
            "tiempo_busqueda_ms": resultado_rag.tiempo_busqueda_ms,
        }

    # ================================================================
    # EXPORTAR PARA VISUALIZACIÓN
    # ================================================================

    def exportar_datos_visualizacion(self, query: str) -> Dict:
        """
        Exporta todos los datos necesarios para la visualización HTML/JS.

        Returns:
            Dict con embeddings, similitudes, documentos, etc.
            Listo para serializar a JSON y consumir desde JavaScript.
        """
        resultado = self.buscar(query)

        # Preparar datos de embeddings para visualización
        todos_embeddings = []
        if self.embeddings_almacenados is not None:
            for i, emb in enumerate(self.embeddings_almacenados):
                doc = self.base_conocimiento.obtener_todos()[i]
                todos_embeddings.append({
                    "id": doc.id,
                    "categoria": doc.categoria,
                    "descripcion": doc.descripcion_natural[:50],
                    # Solo primeras 20 dims para visualización
                    "embedding_preview": emb[:20].tolist(),
                })

        return {
            "query": query,
            "query_embedding_preview": (
                resultado.embeddings_query[:20].tolist()
                if resultado.embeddings_query is not None else []
            ),
            "resultados": resultado.documentos_recuperados,
            "todos_documentos": todos_embeddings,
            "tiempo_busqueda_ms": resultado.tiempo_busqueda_ms,
            "contexto_generado": resultado.contexto_generado,
            "estadisticas_base": self.base_conocimiento.obtener_estadisticas()
            if self.base_conocimiento else {},
        }
