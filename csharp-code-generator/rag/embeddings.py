"""
============================================================================
rag/embeddings.py — Generador de Embeddings para Código C#
============================================================================

PROPÓSITO EDUCATIVO:
    Este módulo convierte texto (código C# + descripciones) en vectores
    numéricos (embeddings) que permiten búsqueda por similitud semántica.

¿QUÉ ES UN EMBEDDING?
    Un embedding es una representación de un texto como un vector de
    números reales de dimensión fija (ej: 256 dimensiones).

    Textos SIMILARES tienen vectores CERCANOS en el espacio vectorial.

    Ejemplo (simplificado a 3D):
    - "crear endpoint REST" → [0.8, 0.2, 0.5]
    - "controller API HTTP"  → [0.7, 0.3, 0.4]  ← CERCANO
    - "conexión a base de datos" → [0.1, 0.9, 0.3]  ← LEJANO

¿POR QUÉ NO USAR BÚSQUEDA POR PALABRAS CLAVE?
    La búsqueda por keywords fallaría en estos casos:
    - Query: "crear endpoint REST" → no matchea "Controller" por keywords
    - Pero semánticamente, un Controller ES un endpoint REST

    Los embeddings capturan SIGNIFICADO, no solo coincidencia de palabras.

EMBEDDINGS PARA CÓDIGO VS TEXTO:
    Los embeddings de código son más difíciles porque:
    1. El código mezcla lenguaje natural (comentarios) con sintaxis
    2. Los nombres de variables son arbitrarios pero informativos
    3. La estructura (indentación, llaves) lleva significado
    4. Abreviaciones son comunes (dto, svc, repo, ctrl)
============================================================================
"""

import numpy as np
import re
from typing import List, Dict, Optional
from collections import Counter
import math


class GeneradorEmbeddings:
    """
    Genera embeddings para texto y código C# usando TF-IDF + hashing.

    NOTA EDUCATIVA — ELECCIÓN DE MÉTODO:
        En producción, se usarían modelos pre-entrenados como:
        - CodeBERT (Microsoft): embeddings específicos para código
        - text-embedding-ada-002 (OpenAI): embeddings generales
        - Sentence-BERT: embeddings de oraciones

        Aquí implementamos TF-IDF + hashing por razones educativas:
        1. No requiere GPU ni modelos grandes
        2. Es completamente transparente (podemos ver qué mide)
        3. Funciona razonablemente bien para nuestro caso de uso
        4. Es rápido y determinístico

    TF-IDF EXPLICADO:
        TF (Term Frequency): ¿Cuántas veces aparece la palabra en ESTE documento?
        IDF (Inverse Document Frequency): ¿Qué tan RARA es la palabra en TODOS los documentos?

        TF-IDF = TF × IDF

        Palabras como "public" tienen IDF bajo (aparecen en todos lados → poco informativas)
        Palabras como "Repository" tienen IDF alto (solo en algunos docs → muy informativas)
    """

    def __init__(self, dimension: int = 256):
        """
        Inicializa el generador de embeddings.

        Args:
            dimension: Dimensión de los vectores resultantes.
                       Debe coincidir con la dimensión usada en FAISS.
        """
        self.dimension = dimension
        self.vocabulario: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.num_documentos = 0
        self._construido = False

        # Palabras de C# con pesos especiales
        # (algunas palabras son más informativas que otras para búsqueda)
        self.pesos_csharp = {
            "controller": 2.0, "service": 2.0, "repository": 2.0,
            "dto": 1.5, "model": 1.5, "interface": 1.5,
            "async": 1.3, "await": 1.3, "task": 1.3,
            "linq": 1.5, "query": 1.3, "where": 1.2,
            "httpget": 2.0, "httppost": 2.0, "httpput": 2.0,
            "httpdelete": 2.0, "api": 1.5, "rest": 1.5,
            "endpoint": 1.5, "crud": 1.5, "validation": 1.3,
            "middleware": 1.5, "authentication": 1.5,
            "generic": 1.3, "pagination": 1.3,
            "entity": 1.3, "framework": 1.2,
        }

    def construir(self, textos: List[str]):
        """
        Construye el vocabulario y calcula IDF desde un corpus.

        NOTA EDUCATIVA:
            Este paso es análogo a "entrenar" el modelo de embeddings.
            Analizamos todos los documentos para saber:
            1. Qué términos existen (vocabulario)
            2. Qué tan comunes son (IDF)

            Un término que aparece en TODOS los documentos (como "public")
            tiene IDF ≈ 0, es decir, NO ayuda a distinguir documentos.
            Un término raro (como "SemaphoreSlim") tiene IDF alto.
        """
        print("🔧 Construyendo modelo de embeddings...")

        self.num_documentos = len(textos)
        doc_frecuencias: Counter = Counter()

        # Paso 1: Contar en cuántos documentos aparece cada término
        for texto in textos:
            terminos = set(self._preprocesar(texto))
            doc_frecuencias.update(terminos)

        # Paso 2: Calcular IDF
        for termino, df in doc_frecuencias.items():
            # IDF = log(N / df) + 1 (suavizado para evitar log(0))
            self.idf[termino] = math.log(self.num_documentos / (df + 1)) + 1

        # Paso 3: Crear vocabulario (mapeo término → índice)
        self.vocabulario = {
            termino: idx
            for idx, termino in enumerate(sorted(doc_frecuencias.keys()))
        }

        self._construido = True
        print(f"✅ Modelo de embeddings construido:")
        print(f"   Vocabulario: {len(self.vocabulario)} términos")
        print(f"   Documentos: {self.num_documentos}")
        print(f"   Dimensión de embeddings: {self.dimension}")

    def generar_embedding(self, texto: str) -> np.ndarray:
        """
        Genera un embedding para un texto dado.

        PROCESO:
        1. Preprocesar texto → lista de términos
        2. Calcular TF-IDF para cada término
        3. Aplicar pesos especiales de C#
        4. Reducir a dimensión fija con hashing trick

        NOTA EDUCATIVA — HASHING TRICK:
            Como el vocabulario puede ser muy grande pero queremos
            embeddings de dimensión fija (256), usamos una función
            hash para mapear cada término a una posición en el vector.

            Esto es como "comprimir" el vector sparse de TF-IDF
            a un vector denso de tamaño fijo.

            Ventaja: dimensión fija sin importar el tamaño del vocabulario
            Desventaja: posibles colisiones (dos términos en la misma posición)
        """
        terminos = self._preprocesar(texto)

        if not terminos:
            return np.zeros(self.dimension, dtype=np.float32)

        # Calcular TF
        tf = Counter(terminos)
        max_tf = max(tf.values()) if tf else 1

        # Vector de embedding (inicializado en ceros)
        embedding = np.zeros(self.dimension, dtype=np.float32)

        for termino, frecuencia in tf.items():
            # TF normalizado
            tf_norm = 0.5 + 0.5 * (frecuencia / max_tf)

            # IDF (usar 1.0 si el término no está en el vocabulario)
            idf_valor = self.idf.get(termino, 1.0)

            # Peso especial de C#
            peso_csharp = self.pesos_csharp.get(termino.lower(), 1.0)

            # TF-IDF con peso
            tfidf = tf_norm * idf_valor * peso_csharp

            # Hashing trick: mapear término a posición(es) en el vector
            hash1 = hash(termino) % self.dimension
            hash2 = hash(termino + "_2") % self.dimension

            # Usar signo alterno para reducir colisiones
            signo = 1 if hash(termino) % 2 == 0 else -1

            embedding[hash1] += tfidf * signo
            embedding[hash2] += tfidf * 0.5 * (-signo)

        # Normalizar L2 (longitud unitaria)
        norma = np.linalg.norm(embedding)
        if norma > 0:
            embedding = embedding / norma

        return embedding

    def generar_embeddings_batch(self, textos: List[str]) -> np.ndarray:
        """
        Genera embeddings para múltiples textos.

        Returns:
            Matriz numpy de forma (num_textos, dimension)
        """
        embeddings = np.zeros((len(textos), self.dimension), dtype=np.float32)

        for i, texto in enumerate(textos):
            embeddings[i] = self.generar_embedding(texto)

        return embeddings

    def _preprocesar(self, texto: str) -> List[str]:
        """
        Preprocesa texto para extracción de términos.

        NOTA EDUCATIVA:
            El preprocesamiento para CÓDIGO es diferente que para texto:
            1. Dividimos PascalCase (GetAllProducts → get all products)
            2. Mantenemos tokens técnicos (async, await, linq)
            3. Eliminamos puntuación de código ({, }, ;, etc.)
            4. Convertimos a minúsculas para matching flexible

            Pero NO eliminamos "stopwords" como en NLP clásico,
            porque en código, palabras como "return", "new", "this"
            son semánticamente importantes.
        """
        # Paso 1: Dividir PascalCase
        texto = re.sub(r'([A-Z][a-z]+)', r' \1 ', texto)
        texto = re.sub(r'([A-Z]{2,})', r' \1 ', texto)

        # Paso 2: Eliminar puntuación de código (pero preservar keywords)
        texto = re.sub(r'[{}()\[\];:,.<>!=+\-*/&|^~?@#\\\'\"]+', ' ', texto)

        # Paso 3: Convertir a minúsculas y dividir en tokens
        terminos = texto.lower().split()

        # Paso 4: Filtrar tokens muy cortos (excepto keywords de C#)
        keywords_cortas = {'if', 'do', 'in', 'as', 'is'}
        terminos = [
            t for t in terminos
            if len(t) >= 3 or t in keywords_cortas
        ]

        return terminos

    def similitud_coseno(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos vectores.

        NOTA EDUCATIVA:
            La similitud coseno mide el ángulo entre dos vectores:
            - 1.0 = vectores idénticos (misma dirección)
            - 0.0 = vectores ortogonales (sin relación)
            - -1.0 = vectores opuestos

            Usamos coseno en lugar de distancia euclidiana porque
            nos importa la DIRECCIÓN del vector (qué términos son
            importantes) más que su MAGNITUD (cuántas palabras tiene).
        """
        norma_a = np.linalg.norm(vec_a)
        norma_b = np.linalg.norm(vec_b)

        if norma_a == 0 or norma_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norma_a * norma_b))
