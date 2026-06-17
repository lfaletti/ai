"""
============================================================================
modelo/tokenizador_csharp.py — Tokenizador Especializado para C#
============================================================================

PROPÓSITO EDUCATIVO:
    Este módulo implementa un tokenizador ESPECIALIZADO para código C#.
    Es fundamentalmente diferente de un tokenizador de texto natural.

¿POR QUÉ NO USAR EL MISMO TOKENIZADOR QUE GPT?
    GPT usa BPE (Byte Pair Encoding) entrenado en texto natural.
    Esto causa problemas con código C#:

    1. OPERADORES COMPUESTOS: "=>" se tokeniza como "=" + ">" en BPE genérico.
       Pero "=>" es un ÚNICO token semántico (operador lambda en C#).

    2. PASCAL CASE: "GetAllProductsAsync" se rompe en sub-tokens aleatorios.
       Nosotros lo dividimos en ["Get", "All", "Products", "Async"] porque
       cada parte tiene significado semántico.

    3. GENÉRICOS: "List<string>" se tokeniza como "List", "<", "string", ">"
       en BPE genérico. Pero "<string>" es parte del TIPO, no una comparación.

    4. INDENTACIÓN: Los espacios de indentación en código son SIGNIFICATIVOS
       (indican nivel de anidamiento). En texto natural, los espacios son
       triviales.

    5. COMENTARIOS XML: "/// <summary>" es un token especial de documentación
       en C#, no HTML genérico.

VOCABULARIO DE CÓDIGO VS TEXTO NATURAL:
    - Texto natural: ~50,000 tokens (palabras comunes + subpalabras)
    - Código C#: ~30,000 tokens PERO con distribución MUY diferente:
      * Keywords: "public", "class", "async" son EXTREMADAMENTE frecuentes
      * Delimitadores: "{", "}", "(", ")" aparecen cientos de veces
      * Tipos: "int", "string", "List<T>" son tokens de alta frecuencia
      * Identificadores: nombres de variables/métodos son de baja frecuencia
        pero alta importancia semántica
============================================================================
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, OrderedDict


class TokenizadorCSharp:
    """
    Tokenizador especializado para código C# con soporte FIM.

    CARACTERÍSTICAS ESPECIALES:
    1. Reconoce operadores compuestos de C# como tokens únicos
    2. Divide PascalCase/camelCase en sub-tokens significativos
    3. Maneja tipos genéricos como unidades semánticas
    4. Soporta Fill-in-the-Middle (FIM) con tokens especiales
    5. Preserva indentación como tokens de estructura
    """

    # ================================================================
    # Tokens especiales para el modelo
    # ================================================================

    # Tokens de control del modelo
    PAD_TOKEN = "<PAD>"      # Relleno para hacer batches del mismo largo
    UNK_TOKEN = "<UNK>"      # Token desconocido (no está en vocabulario)
    BOS_TOKEN = "<BOS>"      # Beginning Of Sequence (inicio de archivo)
    EOS_TOKEN = "<EOS>"      # End Of Sequence (fin de archivo)

    # Tokens FIM (Fill-in-the-Middle)
    # ================================
    # FIM permite al modelo completar código EN MEDIO de un archivo,
    # no solo al final. Esto es CRUCIAL para un copilot útil.
    #
    # Ejemplo:
    #   Código original: "public int Sumar(int a, int b) { return ___; }"
    #   Con FIM:
    #     PREFIX: "public int Sumar(int a, int b) { return "
    #     SUFFIX: "; }"
    #     MIDDLE: "a + b"  ← esto es lo que el modelo debe generar
    #
    # El modelo recibe: <FIM_PREFIX>...prefix...<FIM_SUFFIX>...suffix...<FIM_MIDDLE>
    # Y debe generar los tokens del middle.
    FIM_PREFIX = "<FIM_PREFIX>"
    FIM_SUFFIX = "<FIM_SUFFIX>"
    FIM_MIDDLE = "<FIM_MIDDLE>"

    # Token de nueva línea (explícito para que el modelo lo aprenda)
    NEWLINE_TOKEN = "<NL>"

    # Tokens de indentación (el modelo debe aprender a indentar correctamente)
    INDENT_TOKEN = "<INDENT>"    # 4 espacios o 1 tab
    DEDENT_TOKEN = "<DEDENT>"    # Reducción de indentación

    # ================================================================
    # Keywords y operadores de C#
    # ================================================================

    # Keywords de C# que deben ser tokens individuales
    CSHARP_KEYWORDS = {
        # Modificadores de acceso
        "public", "private", "protected", "internal",
        # Modificadores de clase/método
        "static", "abstract", "virtual", "override", "sealed",
        "readonly", "const", "volatile",
        # Tipos y declaraciones
        "class", "struct", "interface", "enum", "record",
        "namespace", "using", "var", "dynamic",
        # Control de flujo
        "if", "else", "switch", "case", "default", "break",
        "for", "foreach", "while", "do", "continue",
        "return", "yield", "throw", "try", "catch", "finally",
        # Async
        "async", "await", "Task",
        # LINQ keywords
        "from", "where", "select", "orderby", "group",
        "join", "into", "let", "ascending", "descending",
        "on", "equals",
        # Tipos primitivos
        "int", "string", "bool", "decimal", "double", "float",
        "long", "short", "byte", "char", "void", "object",
        # Valores literales
        "true", "false", "null", "this", "base", "new",
        # Operadores lógicos
        "is", "as", "in", "out", "ref", "params",
        # Otros
        "get", "set", "init", "value", "nameof", "typeof",
        "when", "with",
    }

    # Operadores compuestos que deben ser tokens ÚNICOS
    # (ordenados de más largo a más corto para matching correcto)
    OPERADORES_COMPUESTOS = [
        "=>",   # Lambda (MUY importante en C#)
        "??=",  # Null-coalescing assignment
        "??",   # Null-coalescing
        "?.",   # Null-conditional
        "?..",  # Null-conditional range
        "?..",  # Null-conditional element access
        "==",   # Igualdad
        "!=",   # Desigualdad
        "<=",   # Menor o igual
        ">=",   # Mayor o igual
        "&&",   # AND lógico
        "||",   # OR lógico
        "++",   # Incremento
        "--",   # Decremento
        "+=",   # Asignación con suma
        "-=",   # Asignación con resta
        "*=",   # Asignación con multiplicación
        "/=",   # Asignación con división
        "<<",   # Shift izquierdo
        ">>",   # Shift derecho
        "::",   # Scope resolution
        "..",   # Range operator (C# 8+)
    ]

    # Operadores simples
    OPERADORES_SIMPLES = set("{}()[]<>;:,.=+-*/%&|^!~?@#")

    # Tipos genéricos comunes en C# (se tratan como unidades)
    TIPOS_GENERICOS_COMUNES = [
        "List<", "Dictionary<", "IEnumerable<", "IQueryable<",
        "Task<", "ActionResult<", "IList<", "ICollection<",
        "HashSet<", "Queue<", "Stack<", "IReadOnlyList<",
        "Func<", "Action<", "Expression<", "Nullable<",
        "ILogger<", "DbSet<",
    ]

    # Atributos comunes de ASP.NET Core
    ATRIBUTOS_ASPNET = [
        "[ApiController]", "[HttpGet]", "[HttpPost]",
        "[HttpPut]", "[HttpDelete]", "[HttpPatch]",
        "[FromBody]", "[FromQuery]", "[FromRoute]",
        "[Authorize]", "[AllowAnonymous]",
        "[ProducesResponseType", "[Route(",
    ]

    def __init__(self, tam_vocab_max: int = 8000):
        """
        Inicializa el tokenizador con vocabulario vacío.

        NOTA EDUCATIVA:
            tam_vocab_max controla el tamaño máximo del vocabulario.
            - Muy pequeño (1000): muchos tokens <UNK>, mala calidad
            - Muy grande (50000): modelo más grande, más memoria
            - Punto dulce para código C#: 5000-10000 tokens

            Los modelos de producción (Copilot) usan ~50,000 tokens
            pero incluyen múltiples lenguajes.
        """
        self.tam_vocab_max = tam_vocab_max

        # Mapeos token ↔ id
        self.token_a_id: Dict[str, int] = {}
        self.id_a_token: Dict[int, str] = {}

        # Frecuencias de tokens (para construir vocabulario)
        self.frecuencias: Counter = Counter()

        # Flag para saber si el vocabulario está construido
        self._vocabulario_construido = False

        # Pre-registrar tokens especiales (siempre tienen los IDs más bajos)
        self._registrar_tokens_especiales()

    def _registrar_tokens_especiales(self):
        """
        Registra tokens especiales con IDs fijos.

        NOTA EDUCATIVA:
            Los tokens especiales SIEMPRE tienen los mismos IDs
            sin importar el dataset. Esto es importante porque:
            1. PAD (0) se usa para masking en el modelo
            2. Los tokens FIM deben ser reconocibles
            3. El modelo aprende qué hacer con cada token especial
        """
        tokens_especiales = [
            self.PAD_TOKEN,     # ID 0
            self.UNK_TOKEN,     # ID 1
            self.BOS_TOKEN,     # ID 2
            self.EOS_TOKEN,     # ID 3
            self.FIM_PREFIX,    # ID 4
            self.FIM_SUFFIX,    # ID 5
            self.FIM_MIDDLE,    # ID 6
            self.NEWLINE_TOKEN, # ID 7
            self.INDENT_TOKEN,  # ID 8
            self.DEDENT_TOKEN,  # ID 9
        ]

        for idx, token in enumerate(tokens_especiales):
            self.token_a_id[token] = idx
            self.id_a_token[idx] = token

    # ================================================================
    # TOKENIZACIÓN: Convertir código C# en tokens
    # ================================================================

    def tokenizar(self, codigo: str) -> List[str]:
        """
        Convierte código C# en una lista de tokens.

        PROCESO DE TOKENIZACIÓN (específico para C#):
        1. Dividir en líneas (preservar estructura)
        2. Extraer indentación de cada línea
        3. Tokenizar cada línea:
           a. Reconocer strings y comentarios (no tokenizar su interior)
           b. Reconocer operadores compuestos como tokens únicos
           c. Dividir PascalCase/camelCase en sub-tokens
           d. Reconocer tipos genéricos
           e. Reconocer keywords de C#

        EJEMPLO:
            Input:  "public async Task<int> GetTotal()"
            Output: ["public", "async", "Task<", "int", ">",
                     "Get", "Total", "(", ")"]
        """
        tokens = []
        lineas = codigo.split('\n')

        for linea in lineas:
            if not linea.strip():
                # Línea vacía → token de nueva línea
                tokens.append(self.NEWLINE_TOKEN)
                continue

            # Paso 1: Extraer y tokenizar indentación
            espacios = len(linea) - len(linea.lstrip())
            nivel_indent = espacios // 4
            for _ in range(nivel_indent):
                tokens.append(self.INDENT_TOKEN)

            # Paso 2: Tokenizar el contenido de la línea
            contenido = linea.strip()
            tokens_linea = self._tokenizar_linea(contenido)
            tokens.extend(tokens_linea)

            # Paso 3: Agregar nueva línea al final
            tokens.append(self.NEWLINE_TOKEN)

        return tokens

    def _tokenizar_linea(self, linea: str) -> List[str]:
        """
        Tokeniza una línea individual de código C#.

        NOTA EDUCATIVA:
            Esta función es el corazón del tokenizador. Debe manejar
            múltiples tipos de "sub-lenguajes" dentro de C#:
            - Código normal (keywords, identificadores, operadores)
            - Strings literales ("texto" o $"interpolado {var}")
            - Comentarios (// línea, /* bloque */, /// XML)
            - Atributos ([HttpGet], [ApiController])
            - Directivas de preprocesador (#region, #if)
        """
        tokens = []
        i = 0

        while i < len(linea):
            c = linea[i]

            # Saltar espacios (ya manejamos indentación arriba)
            if c == ' ' or c == '\t':
                i += 1
                continue

            # ---- Comentarios de línea (// ...) ----
            if i + 1 < len(linea) and linea[i:i+2] == '//':
                comentario = linea[i:]
                # Comentarios XML (///) son especiales en C#
                if comentario.startswith('///'):
                    tokens.append('///')
                    tokens.extend(self._tokenizar_texto(comentario[3:].strip()))
                else:
                    tokens.append('//')
                    tokens.extend(self._tokenizar_texto(comentario[2:].strip()))
                break  # El resto de la línea es comentario

            # ---- Strings literales ----
            if c == '"' or (c == '$' and i + 1 < len(linea) and linea[i+1] == '"'):
                string_token, avance = self._extraer_string(linea, i)
                tokens.append(string_token)
                i += avance
                continue

            # ---- Caracteres literales ----
            if c == '\'' and i + 2 < len(linea):
                fin = linea.find("'", i + 1)
                if fin != -1:
                    tokens.append(linea[i:fin+1])
                    i = fin + 1
                    continue

            # ---- Operadores compuestos (verificar ANTES que simples) ----
            operador_encontrado = False
            for op in self.OPERADORES_COMPUESTOS:
                if linea[i:].startswith(op):
                    tokens.append(op)
                    i += len(op)
                    operador_encontrado = True
                    break

            if operador_encontrado:
                continue

            # ---- Operadores simples y delimitadores ----
            if c in self.OPERADORES_SIMPLES:
                tokens.append(c)
                i += 1
                continue

            # ---- Números ----
            if c.isdigit():
                num, avance = self._extraer_numero(linea, i)
                tokens.append(num)
                i += avance
                continue

            # ---- Identificadores y keywords ----
            if c.isalpha() or c == '_':
                ident, avance = self._extraer_identificador(linea, i)
                i += avance

                # ¿Es una keyword de C#?
                if ident in self.CSHARP_KEYWORDS:
                    tokens.append(ident)
                else:
                    # Dividir PascalCase en sub-tokens
                    sub_tokens = self._dividir_pascal_case(ident)
                    tokens.extend(sub_tokens)
                continue

            # ---- Carácter no reconocido ----
            tokens.append(c)
            i += 1

        return tokens

    def _dividir_pascal_case(self, identificador: str) -> List[str]:
        """
        Divide un identificador PascalCase en sub-tokens significativos.

        NOTA EDUCATIVA — POR QUÉ DIVIDIR PASCAL CASE:
            "GetAllProductsAsync" → ["Get", "All", "Products", "Async"]

            Esto es importante porque:
            1. "Get" aparece en muchos métodos → el modelo aprende que
               "Get" = operación de lectura
            2. "Async" al final → el modelo aprende que el método es asíncrono
            3. "Products" → el modelo identifica la entidad

            Si no dividimos, "GetAllProductsAsync" sería un token raro
            que el modelo vería pocas veces. Dividido, cada parte es
            un token frecuente con semántica clara.

        CONTRASTE CON TEXTO NATURAL:
            En español, "supermercado" NO se divide en "super" + "mercado"
            en la mayoría de tokenizadores. Pero en código, la división
            es SIEMPRE informativa.
        """
        if len(identificador) <= 2:
            return [identificador]

        # Patrón para dividir en boundaries de PascalCase/camelCase
        partes = re.findall(
            r'[A-Z](?:[A-Z]*(?=[A-Z][a-z])|[a-z]*)|[a-z]+|[0-9]+|_+',
            identificador
        )

        if not partes:
            return [identificador]

        # Filtrar partes vacías y underscores solos
        return [p for p in partes if p and p != '_'] or [identificador]

    def _extraer_string(self, linea: str, inicio: int) -> Tuple[str, int]:
        """
        Extrae un string literal completo, incluyendo interpolados.

        NOTA EDUCATIVA:
            C# tiene MUCHOS tipos de strings:
            - "normal"
            - @"verbatim con \\n literal"
            - $"interpolado {variable}"
            - $@"interpolado verbatim"
            - \"\"\"raw string (C# 11)\"\"\"

            Cada uno tiene reglas de escape diferentes.
            Nuestro tokenizador simplifica esto tratando todo el
            string como un solo token. En producción, se tokenizaría
            el interior del string por separado.
        """
        i = inicio
        prefijo = ""

        # Detectar prefijos de string
        if linea[i] == '$':
            prefijo = "$"
            i += 1
        if i < len(linea) and linea[i] == '@':
            prefijo += "@"
            i += 1

        if i >= len(linea) or linea[i] != '"':
            return (prefijo, len(prefijo))

        # Buscar el cierre del string
        i += 1  # Saltar la comilla de apertura
        while i < len(linea):
            if linea[i] == '\\' and i + 1 < len(linea):
                i += 2  # Saltar escape
                continue
            if linea[i] == '"':
                i += 1
                break
            i += 1

        string_completo = linea[inicio:i]
        return (string_completo, i - inicio)

    def _extraer_numero(self, linea: str, inicio: int) -> Tuple[str, int]:
        """Extrae un número literal (entero, decimal, hex)."""
        i = inicio
        while i < len(linea) and (linea[i].isdigit() or linea[i] in '.xXfFdDmMlL_'):
            i += 1
        return (linea[inicio:i], i - inicio)

    def _extraer_identificador(self, linea: str, inicio: int) -> Tuple[str, int]:
        """Extrae un identificador (variable, tipo, método, etc.)."""
        i = inicio
        while i < len(linea) and (linea[i].isalnum() or linea[i] == '_'):
            i += 1
        return (linea[inicio:i], i - inicio)

    def _tokenizar_texto(self, texto: str) -> List[str]:
        """Tokeniza texto dentro de comentarios (más simple)."""
        palabras = texto.split()
        return palabras if palabras else []

    # ================================================================
    # VOCABULARIO: Construir el mapeo token → ID
    # ================================================================

    def construir_vocabulario(self, codigos: List[str]):
        """
        Construye el vocabulario a partir de un corpus de código C#.

        PROCESO:
        1. Tokenizar todo el corpus
        2. Contar frecuencias de cada token
        3. Seleccionar los top-N tokens más frecuentes
        4. Asignar IDs secuenciales

        NOTA EDUCATIVA — DISTRIBUCIÓN DE FRECUENCIAS EN CÓDIGO:
            La distribución de tokens en código sigue la ley de Zipf
            (como en texto natural), pero con una "cabeza" más pesada:
            - "{" y "}" pueden ser los tokens más frecuentes
            - "public" es extremadamente común en C#
            - Los nombres de variables son la "cola larga"

            Esta distribución afecta el entrenamiento: el modelo
            verá "{" miles de veces pero "GetAllProductsAsync" quizás
            solo una vez. Por eso dividimos PascalCase.
        """
        print("🔨 Construyendo vocabulario desde corpus de código C#...")

        # Contar frecuencias de todos los tokens
        self.frecuencias = Counter()

        for codigo in codigos:
            tokens = self.tokenizar(codigo)
            self.frecuencias.update(tokens)

        # Mantener tokens especiales (ya registrados)
        siguiente_id = len(self.token_a_id)

        # Agregar keywords de C# primero (siempre deben estar en el vocabulario)
        for kw in sorted(self.CSHARP_KEYWORDS):
            if kw not in self.token_a_id:
                self.token_a_id[kw] = siguiente_id
                self.id_a_token[siguiente_id] = kw
                siguiente_id += 1

        # Agregar operadores compuestos
        for op in self.OPERADORES_COMPUESTOS:
            if op not in self.token_a_id:
                self.token_a_id[op] = siguiente_id
                self.id_a_token[siguiente_id] = op
                siguiente_id += 1

        # Agregar operadores simples
        for op in sorted(self.OPERADORES_SIMPLES):
            if op not in self.token_a_id:
                self.token_a_id[op] = siguiente_id
                self.id_a_token[siguiente_id] = op
                siguiente_id += 1

        # Agregar tokens más frecuentes hasta llenar el vocabulario
        tokens_por_frecuencia = self.frecuencias.most_common()

        for token, freq in tokens_por_frecuencia:
            if siguiente_id >= self.tam_vocab_max:
                break
            if token not in self.token_a_id:
                self.token_a_id[token] = siguiente_id
                self.id_a_token[siguiente_id] = token
                siguiente_id += 1

        self._vocabulario_construido = True

        print(f"✅ Vocabulario construido: {len(self.token_a_id)} tokens")
        print(f"   Keywords C#: {len(self.CSHARP_KEYWORDS)}")
        print(f"   Operadores: {len(self.OPERADORES_COMPUESTOS) + len(self.OPERADORES_SIMPLES)}")
        print(f"   Tokens de corpus: {siguiente_id - len(self.CSHARP_KEYWORDS)}")

    # ================================================================
    # CODIFICACIÓN/DECODIFICACIÓN: Token ↔ ID
    # ================================================================

    def codificar(self, codigo: str, agregar_especiales: bool = True) -> List[int]:
        """
        Convierte código C# en una secuencia de IDs numéricos.

        NOTA EDUCATIVA:
            Los modelos neuronales no entienden texto — solo números.
            Esta función transforma:
            "public class Product" → [45, 23, 102, 87, 67]

            Cada ID es un índice en la tabla de embeddings del modelo.
            El modelo aprende un vector denso para cada ID.
        """
        tokens = self.tokenizar(codigo)
        ids = []

        if agregar_especiales:
            ids.append(self.token_a_id[self.BOS_TOKEN])

        for token in tokens:
            if token in self.token_a_id:
                ids.append(self.token_a_id[token])
            else:
                ids.append(self.token_a_id[self.UNK_TOKEN])

        if agregar_especiales:
            ids.append(self.token_a_id[self.EOS_TOKEN])

        return ids

    def decodificar(self, ids: List[int]) -> str:
        """
        Convierte una secuencia de IDs de vuelta a código C#.

        NOTA EDUCATIVA:
            Esta es la operación inversa de codificar().
            Cuando el modelo genera IDs, los decodificamos para
            obtener código legible. El desafío es reconstruir:
            - Indentación correcta
            - Espacios entre tokens
            - Saltos de línea apropiados
        """
        tokens = []
        for id_token in ids:
            if id_token in self.id_a_token:
                token = self.id_a_token[id_token]
                # Saltar tokens de control
                if token in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                    continue
                tokens.append(token)
            else:
                tokens.append(self.UNK_TOKEN)

        return self._reconstruir_codigo(tokens)

    def _reconstruir_codigo(self, tokens: List[str]) -> str:
        """
        Reconstruye código C# legible a partir de tokens.

        NOTA EDUCATIVA:
            La reconstrucción no es trivial porque debemos decidir:
            - ¿Agregar espacio entre estos dos tokens?
            - ¿Agregar nueva línea aquí?
            - ¿Cuánta indentación poner?

            Reglas heurísticas para C#:
            - Después de "{" → nueva línea + indentación
            - Antes de "}" → des-indentar
            - Después de ";" → nueva línea
            - No espacio antes de "(", ")", ".", ","
            - Sí espacio antes y después de "=", "=>", "+", etc.
        """
        resultado = []
        nivel_indent = 0

        # Tokens que no necesitan espacio antes
        sin_espacio_antes = {'(', ')', '[', ']', '.', ',', ';', ':', '<', '>'}
        # Tokens que no necesitan espacio después
        sin_espacio_despues = {'(', '[', '.', '<', '!', '~'}

        for i, token in enumerate(tokens):
            if token == self.NEWLINE_TOKEN:
                resultado.append('\n')
                # Agregar indentación
                resultado.append('    ' * nivel_indent)
                continue

            if token == self.INDENT_TOKEN:
                nivel_indent += 1
                continue

            if token == self.DEDENT_TOKEN:
                nivel_indent = max(0, nivel_indent - 1)
                continue

            if token == self.FIM_PREFIX or token == self.FIM_SUFFIX or token == self.FIM_MIDDLE:
                resultado.append(f" {token} ")
                continue

            # Decidir si agregar espacio antes del token
            necesita_espacio = True
            if not resultado or resultado[-1].endswith('\n') or resultado[-1].endswith('    '):
                necesita_espacio = False
            elif token in sin_espacio_antes:
                necesita_espacio = False
            elif i > 0 and tokens[i-1] in sin_espacio_despues:
                necesita_espacio = False

            if necesita_espacio:
                resultado.append(' ')

            resultado.append(token)

        return ''.join(resultado)

    # ================================================================
    # FILL-IN-THE-MIDDLE (FIM)
    # ================================================================

    def preparar_fim(self, codigo: str, posicion_cursor: Optional[int] = None) -> Dict:
        """
        Prepara un ejemplo para Fill-in-the-Middle.

        NOTA EDUCATIVA — FIM:
            FIM es una técnica que permite al modelo completar código
            en CUALQUIER posición, no solo al final. Esto es fundamental
            para un copilot útil porque los desarrolladores frecuentemente
            necesitan completar código en medio de un archivo.

        CÓMO FUNCIONA:
            1. Se divide el código en PREFIX, MIDDLE y SUFFIX
            2. Se reordena como: <FIM_PREFIX>prefix<FIM_SUFFIX>suffix<FIM_MIDDLE>
            3. El modelo genera los tokens del MIDDLE

        ¿POR QUÉ ESTE ORDEN (PREFIX → SUFFIX → MIDDLE)?
            Porque el modelo necesita ver AMBOS contextos (antes y después)
            ANTES de generar. Si pusiera PREFIX → MIDDLE → SUFFIX,
            el modelo no vería el contexto posterior al generar el middle.

        EJEMPLO PRÁCTICO:
            Código: "for (int i = 0; i < items.Count; ___) { ... }"
            El modelo ve el for, la condición, Y el cuerpo del loop
            antes de decidir que el incremento es "i++".
        """
        if posicion_cursor is None:
            # Elegir una posición aleatoria razonable (no al inicio ni al final)
            longitud = len(codigo)
            inicio_rango = max(1, longitud // 4)
            fin_rango = min(longitud - 1, 3 * longitud // 4)
            posicion_cursor = (inicio_rango + fin_rango) // 2

        # Encontrar los límites naturales (líneas completas)
        inicio_linea = codigo.rfind('\n', 0, posicion_cursor) + 1
        fin_linea = codigo.find('\n', posicion_cursor)
        if fin_linea == -1:
            fin_linea = len(codigo)

        # Determinar el "middle" como la línea donde está el cursor
        # (en la práctica sería un span más inteligente)
        # Usamos un rango alrededor del cursor
        tam_middle = min(50, (fin_linea - inicio_linea))
        inicio_middle = max(0, posicion_cursor - tam_middle // 2)
        fin_middle = min(len(codigo), posicion_cursor + tam_middle // 2)

        prefijo = codigo[:inicio_middle]
        middle = codigo[inicio_middle:fin_middle]
        sufijo = codigo[fin_middle:]

        # Crear la secuencia FIM
        secuencia_fim = (
            f"{self.FIM_PREFIX}{prefijo}"
            f"{self.FIM_SUFFIX}{sufijo}"
            f"{self.FIM_MIDDLE}{middle}"
        )

        return {
            "prefijo": prefijo,
            "middle": middle,
            "sufijo": sufijo,
            "secuencia_fim": secuencia_fim,
            "posicion_cursor": posicion_cursor,
        }

    def codificar_fim(self, prefijo: str, sufijo: str, middle: str = "") -> List[int]:
        """
        Codifica una entrada FIM como secuencia de IDs.

        Formato: <FIM_PREFIX> tokens_prefix <FIM_SUFFIX> tokens_suffix <FIM_MIDDLE> [tokens_middle]
        """
        ids = [self.token_a_id[self.FIM_PREFIX]]
        ids.extend(self._tokens_a_ids(self.tokenizar(prefijo)))
        ids.append(self.token_a_id[self.FIM_SUFFIX])
        ids.extend(self._tokens_a_ids(self.tokenizar(sufijo)))
        ids.append(self.token_a_id[self.FIM_MIDDLE])

        if middle:
            ids.extend(self._tokens_a_ids(self.tokenizar(middle)))

        ids.append(self.token_a_id[self.EOS_TOKEN])
        return ids

    def _tokens_a_ids(self, tokens: List[str]) -> List[int]:
        """Convierte lista de tokens a lista de IDs."""
        return [
            self.token_a_id.get(t, self.token_a_id[self.UNK_TOKEN])
            for t in tokens
        ]

    # ================================================================
    # UTILIDADES
    # ================================================================

    @property
    def tam_vocabulario(self) -> int:
        """Tamaño actual del vocabulario."""
        return len(self.token_a_id)

    @property
    def pad_id(self) -> int:
        return self.token_a_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_a_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_a_id[self.EOS_TOKEN]

    def guardar(self, ruta: str):
        """Guarda el vocabulario en un archivo JSON."""
        datos = {
            "token_a_id": self.token_a_id,
            "tam_vocab_max": self.tam_vocab_max,
            "frecuencias": dict(self.frecuencias.most_common(1000)),
        }
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

    def cargar(self, ruta: str):
        """Carga el vocabulario desde un archivo JSON."""
        with open(ruta, 'r', encoding='utf-8') as f:
            datos = json.load(f)

        self.token_a_id = datos["token_a_id"]
        # Reconstruir el mapeo inverso
        self.id_a_token = {int(v): k for k, v in self.token_a_id.items()}
        self.tam_vocab_max = datos.get("tam_vocab_max", 8000)
        self._vocabulario_construido = True

    def obtener_estadisticas_tokenizacion(self, codigo: str) -> Dict:
        """
        Genera estadísticas detalladas de la tokenización de un código.
        Útil para la visualización educativa.
        """
        tokens = self.tokenizar(codigo)

        # Clasificar tokens
        clasificacion = {
            "keywords": [],
            "identificadores": [],
            "operadores": [],
            "delimitadores": [],
            "strings": [],
            "numeros": [],
            "comentarios": [],
            "especiales": [],
        }

        for token in tokens:
            if token in self.CSHARP_KEYWORDS:
                clasificacion["keywords"].append(token)
            elif token in (self.NEWLINE_TOKEN, self.INDENT_TOKEN,
                         self.DEDENT_TOKEN, self.PAD_TOKEN):
                clasificacion["especiales"].append(token)
            elif any(token.startswith(op) for op in ['==', '!=', '=>', '<=', '>=']):
                clasificacion["operadores"].append(token)
            elif token in self.OPERADORES_SIMPLES:
                clasificacion["delimitadores"].append(token)
            elif token.startswith('"') or token.startswith('$"'):
                clasificacion["strings"].append(token)
            elif token.startswith('//'):
                clasificacion["comentarios"].append(token)
            elif token[0:1].isdigit():
                clasificacion["numeros"].append(token)
            else:
                clasificacion["identificadores"].append(token)

        return {
            "total_tokens": len(tokens),
            "tokens": tokens,
            "clasificacion": clasificacion,
            "tokens_unicos": len(set(tokens)),
            "ratio_compresion": len(codigo) / max(1, len(tokens)),
        }
