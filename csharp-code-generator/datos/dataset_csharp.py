"""
============================================================================
datos/dataset_csharp.py — Dataset Sintético de Código C# Profesional
============================================================================

PROPÓSITO EDUCATIVO:
    Este módulo genera un dataset sintético de código C# que imita patrones
    reales de desarrollo profesional. A diferencia de datasets de texto natural
    (Wikipedia, libros), el código tiene propiedades únicas:

    1. ESTRUCTURA JERÁRQUICA: namespace → class → method → statement
       El modelo debe aprender que "}" cierra el bloque correcto.

    2. TIPOS ESTÁTICOS: C# usa tipos explícitos (int, string, List<T>).
       El modelo debe predecir tipos coherentes con el contexto.

    3. CONVENCIONES DE NOMBRADO: PascalCase para clases/métodos,
       camelCase para variables locales. Esto es información semántica
       codificada en la forma del token.

    4. PATRONES ARQUITECTÓNICOS: Controllers, Services, Repositories, DTOs.
       Estos patrones se repiten y el modelo los puede aprender.

    5. INDENTACIÓN SIGNIFICATIVA: Aunque C# usa llaves, la indentación
       correcta es crucial para la legibilidad del código generado.

DIFERENCIA CLAVE VS TEXTO NATURAL:
    En texto natural, "El gato está en el ___" podría completarse con
    muchas palabras. En código C#, "public int GetTotal()" DEBE ser
    seguido por "{" o "=>" — las opciones son mucho más restringidas
    por la gramática del lenguaje.
============================================================================
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================================
# SECCIÓN 1: Estructuras de datos para representar snippets C#
# ============================================================================

@dataclass
class SnippetCSharp:
    """
    Representa un fragmento de código C# con metadatos.

    ¿Por qué metadatos?
        Los modelos de código se benefician de saber el CONTEXTO del snippet:
        - ¿Es un Controller o un Service?
        - ¿Qué patrón arquitectónico sigue?
        - ¿Qué nivel de complejidad tiene?

        Esta información se usa para:
        1. Balancear el dataset (no solo generar "Hello World")
        2. Evaluar si el modelo genera código del tipo correcto
        3. Alimentar el sistema RAG con metadatos de búsqueda
    """
    codigo: str                          # El código C# en sí
    categoria: str                       # "controller", "service", "dto", etc.
    patron: str                          # Patrón arquitectónico principal
    complejidad: int                     # 1-5 (básico a avanzado)
    descripcion: str                     # Descripción en español de qué hace
    palabras_clave: List[str] = field(default_factory=list)  # Tags para búsqueda RAG
    usa_linq: bool = False               # Si usa LINQ (importante para C#)
    usa_async: bool = False              # Si usa async/await
    usa_generics: bool = False           # Si usa tipos genéricos


# ============================================================================
# SECCIÓN 2: Generadores de patrones C# profesionales
# ============================================================================

class GeneradorPatronesCSharp:
    """
    Genera código C# sintético siguiendo patrones profesionales reales.

    NOTA EDUCATIVA:
        Los generadores de código como Copilot aprenden de MILLONES de
        repositorios reales. Aquí simulamos esos patrones con plantillas
        parametrizadas. La diferencia es:

        - Copilot: aprende patrones implícitamente de datos reales
        - Nosotros: codificamos patrones explícitamente como plantillas

        Ambos enfoques producen código que sigue las mismas convenciones,
        pero nuestro dataset es más pequeño y controlado (ideal para
        un modelo educativo de 4 capas).
    """

    # Nombres realistas para generar código variado
    ENTIDADES = [
        "Product", "Customer", "Order", "Invoice", "Payment",
        "User", "Role", "Permission", "Category", "Tag",
        "Employee", "Department", "Project", "Task", "Report",
        "Notification", "Message", "Comment", "Review", "Rating"
    ]

    TIPOS_DATO = [
        "int", "string", "bool", "decimal", "DateTime",
        "double", "float", "long", "Guid", "byte[]"
    ]

    NAMESPACES = [
        "MiApp.Api.Controllers", "MiApp.Core.Services",
        "MiApp.Core.Models", "MiApp.Core.DTOs",
        "MiApp.Infrastructure.Repositories",
        "MiApp.Core.Interfaces", "MiApp.Core.Validators",
        "MiApp.Api.Middleware", "MiApp.Core.Extensions",
        "MiApp.Infrastructure.Data"
    ]

    VERBOS_HTTP = ["HttpGet", "HttpPost", "HttpPut", "HttpDelete", "HttpPatch"]

    def __init__(self, semilla: int = 42):
        """
        Inicializa el generador con una semilla para reproducibilidad.

        ¿Por qué semilla fija?
            En machine learning, queremos resultados reproducibles.
            Si entrenamos dos veces con los mismos datos, queremos
            comparar resultados justos. La semilla controla el
            "azar" del generador de números aleatorios.
        """
        random.seed(semilla)

    # ----------------------------------------------------------------
    # 2.1: Generación de DTOs (Data Transfer Objects)
    # ----------------------------------------------------------------

    def generar_dto(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera un DTO (Data Transfer Object) de C#.

        ¿Qué es un DTO y por qué importa para generadores de código?
            Los DTOs son uno de los patrones MÁS COMUNES en código C#.
            Son clases simples con propiedades, sin lógica de negocio.
            Un generador de código bueno debe poder crear DTOs automáticamente
            a partir de un nombre de entidad.

        EJEMPLO DE USO REAL:
            Desarrollador escribe: "// DTO para producto con precio y nombre"
            Copilot genera: la clase completa con propiedades tipadas
        """
        entidad = entidad or random.choice(self.ENTIDADES)
        num_props = random.randint(3, 8)

        # Seleccionar propiedades relevantes para la entidad
        propiedades = self._generar_propiedades(entidad, num_props)

        # Construir el código del DTO
        props_code = ""
        for nombre, tipo, descripcion in propiedades:
            props_code += f"""
        /// <summary>
        /// {descripcion}
        /// </summary>
        public {tipo} {nombre} {{ get; set; }}
"""

        # Opcionalmente agregar validaciones con DataAnnotations
        usar_validaciones = random.choice([True, False])
        usings = "using System;\nusing System.ComponentModel.DataAnnotations;\n" if usar_validaciones else "using System;\n"

        codigo = f"""{usings}
namespace MiApp.Core.DTOs
{{
    /// <summary>
    /// DTO para transferir datos de {entidad}.
    /// Los DTOs separan la representación externa de la entidad de dominio.
    /// Esto permite versionar la API sin afectar el modelo interno.
    /// </summary>
    public class {entidad}Dto
    {{{props_code}
    }}

    /// <summary>
    /// DTO para crear un nuevo {entidad}.
    /// Separar Create/Update DTOs es una práctica recomendada
    /// porque al crear no necesitamos el Id.
    /// </summary>
    public class Create{entidad}Dto
    {{{self._filtrar_props_creacion(propiedades)}
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="dto",
            patron="DTO",
            complejidad=1,
            descripcion=f"DTO para la entidad {entidad} con propiedades tipadas",
            palabras_clave=[entidad.lower(), "dto", "modelo", "transferencia"],
            usa_generics=False
        )

    # ----------------------------------------------------------------
    # 2.2: Generación de Interfaces de Servicio
    # ----------------------------------------------------------------

    def generar_interfaz_servicio(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera una interfaz de servicio siguiendo el patrón Repository/Service.

        NOTA EDUCATIVA — INTERFACES EN C#:
            Las interfaces definen CONTRATOS. En generadores de código,
            las interfaces son especialmente útiles porque:
            1. Son predecibles (CRUD básico siempre tiene los mismos métodos)
            2. Sirven como "prompt" para generar la implementación
            3. El modelo aprende que I{Nombre}Service → tiene métodos CRUD

        DIFERENCIA VS TEXTO NATURAL:
            En texto, "contrato" es una metáfora. En C#, una interfaz
            ES LITERALMENTE un contrato que el compilador verifica.
            El modelo de código debe aprender esta relación estricta.
        """
        entidad = entidad or random.choice(self.ENTIDADES)
        usa_async = random.choice([True, True, False])  # Más probable async

        if usa_async:
            tipo_retorno = f"Task<{entidad}Dto>"
            tipo_lista = f"Task<IEnumerable<{entidad}Dto>>"
            tipo_void = "Task"
            prefijo = "async "
        else:
            tipo_retorno = f"{entidad}Dto"
            tipo_lista = f"IEnumerable<{entidad}Dto>"
            tipo_void = "void"
            prefijo = ""

        codigo = f"""using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MiApp.Core.DTOs;

namespace MiApp.Core.Interfaces
{{
    /// <summary>
    /// Interfaz del servicio de {entidad}.
    /// Define el contrato que cualquier implementación debe cumplir.
    /// Esto permite inyección de dependencias y testing con mocks.
    /// </summary>
    public interface I{entidad}Service
    {{
        /// <summary>
        /// Obtiene todos los {entidad}s con paginación opcional.
        /// </summary>
        {tipo_lista} GetAll{entidad}sAsync(int page = 1, int pageSize = 10);

        /// <summary>
        /// Obtiene un {entidad} por su identificador único.
        /// </summary>
        {tipo_retorno} Get{entidad}ByIdAsync(int id);

        /// <summary>
        /// Crea un nuevo {entidad} a partir del DTO proporcionado.
        /// </summary>
        {tipo_retorno} Create{entidad}Async(Create{entidad}Dto dto);

        /// <summary>
        /// Actualiza un {entidad} existente.
        /// </summary>
        {tipo_retorno} Update{entidad}Async(int id, {entidad}Dto dto);

        /// <summary>
        /// Elimina un {entidad} por su identificador.
        /// </summary>
        {tipo_void} Delete{entidad}Async(int id);

        /// <summary>
        /// Busca {entidad}s que coincidan con el término de búsqueda.
        /// </summary>
        {tipo_lista} Search{entidad}sAsync(string searchTerm);
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="interfaz",
            patron="Service Interface",
            complejidad=2,
            descripcion=f"Interfaz del servicio de {entidad} con operaciones CRUD",
            palabras_clave=[entidad.lower(), "interfaz", "servicio", "crud", "contrato"],
            usa_async=usa_async,
            usa_generics=True
        )

    # ----------------------------------------------------------------
    # 2.3: Generación de Servicios (Implementación)
    # ----------------------------------------------------------------

    def generar_servicio(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera la implementación de un servicio con lógica de negocio.

        NOTA EDUCATIVA — IMPLEMENTACIÓN DE SERVICIOS:
            Este es el patrón más complejo que el modelo debe aprender.
            Incluye:
            - Inyección de dependencias en el constructor
            - Validaciones de entrada
            - Manejo de excepciones
            - Mapeo entre entidades y DTOs
            - Lógica de negocio condicional

        ¿POR QUÉ ES DIFÍCIL PARA UN MODELO?
            Porque la implementación debe ser COHERENTE con la interfaz.
            Si la interfaz define "GetByIdAsync(int id)", la implementación
            debe tener exactamente esa firma. Este tipo de coherencia
            a larga distancia es el desafío principal de los modelos
            de código.
        """
        entidad = entidad or random.choice(self.ENTIDADES)

        codigo = f"""using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using MiApp.Core.DTOs;
using MiApp.Core.Interfaces;
using MiApp.Core.Models;

namespace MiApp.Core.Services
{{
    /// <summary>
    /// Implementación del servicio de {entidad}.
    /// Contiene la lógica de negocio y orquesta las operaciones
    /// entre el repositorio y los DTOs.
    /// </summary>
    public class {entidad}Service : I{entidad}Service
    {{
        // ----------------------------------------------------------------
        // Dependencias inyectadas via constructor (patrón DI)
        // ----------------------------------------------------------------
        private readonly I{entidad}Repository _repository;
        private readonly ILogger<{entidad}Service> _logger;
        private readonly IMapper _mapper;

        /// <summary>
        /// Constructor con inyección de dependencias.
        /// El contenedor DI de ASP.NET Core resuelve automáticamente
        /// estas dependencias cuando se registran en Startup/Program.
        /// </summary>
        public {entidad}Service(
            I{entidad}Repository repository,
            ILogger<{entidad}Service> logger,
            IMapper mapper)
        {{
            _repository = repository ?? throw new ArgumentNullException(nameof(repository));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _mapper = mapper ?? throw new ArgumentNullException(nameof(mapper));
        }}

        /// <inheritdoc />
        public async Task<IEnumerable<{entidad}Dto>> GetAll{entidad}sAsync(
            int page = 1, int pageSize = 10)
        {{
            _logger.LogInformation(
                "Obteniendo {entidad}s - Página: {{Page}}, Tamaño: {{PageSize}}",
                page, pageSize);

            // LINQ: Skip y Take implementan paginación eficiente.
            // En EF Core, esto se traduce a OFFSET/FETCH en SQL.
            var entidades = await _repository.GetAllAsync();
            var paginadas = entidades
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToList();

            return _mapper.Map<IEnumerable<{entidad}Dto>>(paginadas);
        }}

        /// <inheritdoc />
        public async Task<{entidad}Dto> Get{entidad}ByIdAsync(int id)
        {{
            if (id <= 0)
                throw new ArgumentException("El id debe ser positivo", nameof(id));

            var entidad = await _repository.GetByIdAsync(id);

            if (entidad == null)
            {{
                _logger.LogWarning("{entidad} con id {{Id}} no encontrado", id);
                throw new KeyNotFoundException($"{entidad} con id {{id}} no existe");
            }}

            return _mapper.Map<{entidad}Dto>(entidad);
        }}

        /// <inheritdoc />
        public async Task<{entidad}Dto> Create{entidad}Async(Create{entidad}Dto dto)
        {{
            if (dto == null)
                throw new ArgumentNullException(nameof(dto));

            _logger.LogInformation("Creando nuevo {entidad}");

            var entidad = _mapper.Map<{entidad}>(dto);
            entidad.FechaCreacion = DateTime.UtcNow;

            var creada = await _repository.AddAsync(entidad);
            await _repository.SaveChangesAsync();

            _logger.LogInformation("{entidad} creado con id {{Id}}", creada.Id);
            return _mapper.Map<{entidad}Dto>(creada);
        }}

        /// <inheritdoc />
        public async Task<{entidad}Dto> Update{entidad}Async(int id, {entidad}Dto dto)
        {{
            var existente = await _repository.GetByIdAsync(id)
                ?? throw new KeyNotFoundException($"{entidad} con id {{id}} no existe");

            _mapper.Map(dto, existente);
            existente.FechaModificacion = DateTime.UtcNow;

            _repository.Update(existente);
            await _repository.SaveChangesAsync();

            return _mapper.Map<{entidad}Dto>(existente);
        }}

        /// <inheritdoc />
        public async Task Delete{entidad}Async(int id)
        {{
            var entidad = await _repository.GetByIdAsync(id)
                ?? throw new KeyNotFoundException($"{entidad} con id {{id}} no existe");

            _repository.Remove(entidad);
            await _repository.SaveChangesAsync();

            _logger.LogInformation("{entidad} con id {{Id}} eliminado", id);
        }}

        /// <inheritdoc />
        public async Task<IEnumerable<{entidad}Dto>> Search{entidad}sAsync(string searchTerm)
        {{
            if (string.IsNullOrWhiteSpace(searchTerm))
                return Enumerable.Empty<{entidad}Dto>();

            // LINQ con expresiones lambda para filtrado flexible
            var resultados = await _repository.FindAsync(
                e => e.Nombre.Contains(searchTerm, StringComparison.OrdinalIgnoreCase));

            return _mapper.Map<IEnumerable<{entidad}Dto>>(resultados);
        }}
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="servicio",
            patron="Service Implementation",
            complejidad=3,
            descripcion=f"Implementación del servicio de {entidad} con CRUD completo",
            palabras_clave=[entidad.lower(), "servicio", "implementacion", "crud", "negocio", "async"],
            usa_linq=True,
            usa_async=True,
            usa_generics=True
        )

    # ----------------------------------------------------------------
    # 2.4: Generación de Controllers API REST
    # ----------------------------------------------------------------

    def generar_controller(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera un Controller de API REST con endpoints CRUD.

        NOTA EDUCATIVA — CONTROLLERS Y GENERACIÓN DE CÓDIGO:
            Los Controllers son el caso de uso PERFECTO para generadores
            de código porque:
            1. Siguen un patrón muy predecible (CRUD)
            2. El routing se infiere del nombre de la entidad
            3. Las respuestas HTTP son estándar (200, 201, 404, etc.)

        CASO DE USO RAG:
            Cuando un desarrollador escribe "crear endpoint API REST para productos",
            el sistema RAG busca controllers existentes similares y los usa
            como contexto para generar el nuevo controller. Esto es MÁS EFECTIVO
            que generar de cero porque mantiene la consistencia del proyecto.
        """
        entidad = entidad or random.choice(self.ENTIDADES)
        ruta = entidad.lower() + "s"

        codigo = f"""using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using MiApp.Core.DTOs;
using MiApp.Core.Interfaces;

namespace MiApp.Api.Controllers
{{
    /// <summary>
    /// Controller REST para gestionar {entidad}s.
    /// Sigue las convenciones RESTful estándar con respuestas HTTP apropiadas.
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Produces("application/json")]
    public class {entidad}sController : ControllerBase
    {{
        private readonly I{entidad}Service _service;
        private readonly ILogger<{entidad}sController> _logger;

        public {entidad}sController(
            I{entidad}Service service,
            ILogger<{entidad}sController> logger)
        {{
            _service = service;
            _logger = logger;
        }}

        /// <summary>
        /// GET api/{ruta}
        /// Obtiene lista paginada de {entidad}s.
        /// </summary>
        [HttpGet]
        [ProducesResponseType(typeof(IEnumerable<{entidad}Dto>), 200)]
        public async Task<ActionResult<IEnumerable<{entidad}Dto>>> GetAll(
            [FromQuery] int page = 1,
            [FromQuery] int pageSize = 10)
        {{
            var resultado = await _service.GetAll{entidad}sAsync(page, pageSize);
            return Ok(resultado);
        }}

        /// <summary>
        /// GET api/{ruta}/{{id}}
        /// Obtiene un {entidad} específico por su ID.
        /// </summary>
        [HttpGet("{{id}}")]
        [ProducesResponseType(typeof({entidad}Dto), 200)]
        [ProducesResponseType(404)]
        public async Task<ActionResult<{entidad}Dto>> GetById(int id)
        {{
            try
            {{
                var resultado = await _service.Get{entidad}ByIdAsync(id);
                return Ok(resultado);
            }}
            catch (KeyNotFoundException)
            {{
                return NotFound(new {{ message = $"{entidad} con id {{id}} no encontrado" }});
            }}
        }}

        /// <summary>
        /// POST api/{ruta}
        /// Crea un nuevo {entidad}.
        /// Retorna 201 Created con la ubicación del nuevo recurso.
        /// </summary>
        [HttpPost]
        [ProducesResponseType(typeof({entidad}Dto), 201)]
        [ProducesResponseType(400)]
        public async Task<ActionResult<{entidad}Dto>> Create(
            [FromBody] Create{entidad}Dto dto)
        {{
            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            var creado = await _service.Create{entidad}Async(dto);
            return CreatedAtAction(
                nameof(GetById),
                new {{ id = creado.Id }},
                creado);
        }}

        /// <summary>
        /// PUT api/{ruta}/{{id}}
        /// Actualiza un {entidad} existente completamente.
        /// </summary>
        [HttpPut("{{id}}")]
        [ProducesResponseType(typeof({entidad}Dto), 200)]
        [ProducesResponseType(404)]
        public async Task<ActionResult<{entidad}Dto>> Update(
            int id, [FromBody] {entidad}Dto dto)
        {{
            try
            {{
                var actualizado = await _service.Update{entidad}Async(id, dto);
                return Ok(actualizado);
            }}
            catch (KeyNotFoundException)
            {{
                return NotFound();
            }}
        }}

        /// <summary>
        /// DELETE api/{ruta}/{{id}}
        /// Elimina un {entidad} por su ID.
        /// </summary>
        [HttpDelete("{{id}}")]
        [ProducesResponseType(204)]
        [ProducesResponseType(404)]
        public async Task<IActionResult> Delete(int id)
        {{
            try
            {{
                await _service.Delete{entidad}Async(id);
                return NoContent();
            }}
            catch (KeyNotFoundException)
            {{
                return NotFound();
            }}
        }}

        /// <summary>
        /// GET api/{ruta}/search?q=término
        /// Busca {entidad}s por término de búsqueda.
        /// </summary>
        [HttpGet("search")]
        [ProducesResponseType(typeof(IEnumerable<{entidad}Dto>), 200)]
        public async Task<ActionResult<IEnumerable<{entidad}Dto>>> Search(
            [FromQuery] string q)
        {{
            var resultados = await _service.Search{entidad}sAsync(q);
            return Ok(resultados);
        }}
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="controller",
            patron="REST Controller",
            complejidad=3,
            descripcion=f"Controller API REST para {entidad} con CRUD completo",
            palabras_clave=[entidad.lower(), "controller", "api", "rest", "endpoint", "crud"],
            usa_async=True,
            usa_generics=True
        )

    # ----------------------------------------------------------------
    # 2.5: Generación de código LINQ avanzado
    # ----------------------------------------------------------------

    def generar_linq_avanzado(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera ejemplos de LINQ avanzado, una característica única de C#.

        NOTA EDUCATIVA — LINQ Y GENERADORES DE CÓDIGO:
            LINQ (Language Integrated Query) es una de las razones por las que
            tokenizar C# es diferente a otros lenguajes:
            - Operadores como "=>" (lambda), "from...select", "where"
            - Encadenamiento de métodos (fluent API)
            - Tipos genéricos anidados: IGrouping<string, List<OrderDto>>

            Un tokenizador genérico rompería "=>" en "=" y ">", perdiendo
            el significado semántico del operador lambda.
        """
        entidad = entidad or random.choice(self.ENTIDADES)

        codigo = f"""using System;
using System.Collections.Generic;
using System.Linq;
using MiApp.Core.Models;

namespace MiApp.Core.Extensions
{{
    /// <summary>
    /// Extensiones LINQ especializadas para consultas de {entidad}.
    /// LINQ es una de las características más poderosas de C# que
    /// permite escribir consultas declarativas sobre colecciones.
    /// </summary>
    public static class {entidad}QueryExtensions
    {{
        /// <summary>
        /// Filtra {entidad}s activos usando expresión lambda.
        /// Lambda (=>) es un operador especial de C# que define
        /// funciones anónimas inline.
        /// </summary>
        public static IQueryable<{entidad}> SoloActivos(
            this IQueryable<{entidad}> query)
        {{
            return query.Where(e => e.EstaActivo && !e.EstaEliminado);
        }}

        /// <summary>
        /// Aplica paginación con Skip/Take.
        /// En EF Core, esto genera SQL con OFFSET/FETCH.
        /// </summary>
        public static IQueryable<T> Paginar<T>(
            this IQueryable<T> query, int pagina, int tamanio)
        {{
            return query
                .Skip((pagina - 1) * tamanio)
                .Take(tamanio);
        }}

        /// <summary>
        /// Agrupa {entidad}s por categoría y calcula estadísticas.
        /// GroupBy + Select con tipo anónimo es un patrón común.
        /// </summary>
        public static IEnumerable<object> EstadisticasPorCategoria(
            this IEnumerable<{entidad}> items)
        {{
            // Sintaxis de método (Method Syntax) - más común en código real
            return items
                .Where(i => i.EstaActivo)
                .GroupBy(i => i.Categoria)
                .Select(g => new
                {{
                    Categoria = g.Key,
                    Total = g.Count(),
                    Promedio = g.Average(i => i.Precio),
                    Maximo = g.Max(i => i.Precio),
                    Minimo = g.Min(i => i.Precio),
                    Items = g.OrderByDescending(i => i.Precio).Take(5)
                }})
                .OrderByDescending(x => x.Total);
        }}

        /// <summary>
        /// Ejemplo de Query Syntax (estilo SQL) vs Method Syntax.
        /// Ambas son equivalentes pero Query Syntax es más legible
        /// para consultas complejas con joins.
        /// </summary>
        public static IEnumerable<{entidad}ResumenDto> ConsultaConJoin(
            this IEnumerable<{entidad}> items,
            IEnumerable<Category> categorias)
        {{
            // Query Syntax - se parece más a SQL
            var resultado =
                from item in items
                join cat in categorias on item.CategoriaId equals cat.Id
                where item.Precio > 0
                orderby item.Nombre ascending
                select new {entidad}ResumenDto
                {{
                    Id = item.Id,
                    Nombre = item.Nombre,
                    CategoriaNombre = cat.Nombre,
                    Precio = item.Precio
                }};

            return resultado.ToList();
        }}

        /// <summary>
        /// LINQ con SelectMany para aplanar colecciones anidadas.
        /// Útil cuando cada {entidad} tiene una lista de sub-items.
        /// </summary>
        public static IEnumerable<string> ObtenerTodasLasEtiquetas(
            this IEnumerable<{entidad}> items)
        {{
            return items
                .Where(i => i.Etiquetas != null)
                .SelectMany(i => i.Etiquetas)
                .Distinct()
                .OrderBy(tag => tag);
        }}

        /// <summary>
        /// Ejemplo de Aggregate para cálculos acumulativos.
        /// Aggregate es el equivalente de reduce() en JavaScript.
        /// </summary>
        public static string GenerarResumen(
            this IEnumerable<{entidad}> items)
        {{
            return items
                .Select(i => $"{{i.Nombre}} (${{i.Precio:C}})")
                .Aggregate((actual, siguiente) => $"{{actual}}, {{siguiente}}");
        }}
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="linq",
            patron="LINQ Extensions",
            complejidad=4,
            descripcion=f"Extensiones LINQ avanzadas para {entidad}",
            palabras_clave=[entidad.lower(), "linq", "query", "lambda", "extension", "filtro"],
            usa_linq=True,
            usa_generics=True
        )

    # ----------------------------------------------------------------
    # 2.6: Generación de patrones async/await
    # ----------------------------------------------------------------

    def generar_patron_async(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera código con patrones async/await de C#.

        NOTA EDUCATIVA — ASYNC EN GENERADORES DE CÓDIGO:
            async/await crea desafíos únicos para los modelos de código:
            1. Cada método async DEBE retornar Task o Task<T>
            2. await solo puede usarse dentro de métodos async
            3. El modelo debe rastrear si estamos en contexto async

            Esto es un ejemplo de DEPENDENCIA A LARGA DISTANCIA:
            la keyword "async" en la firma del método afecta qué
            expresiones son válidas 20-50 líneas después.
        """
        entidad = entidad or random.choice(self.ENTIDADES)

        codigo = f"""using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace MiApp.Infrastructure.ExternalServices
{{
    /// <summary>
    /// Cliente HTTP asíncrono para consumir API externa de {entidad}s.
    /// Demuestra patrones async/await avanzados de C#.
    /// </summary>
    public class {entidad}ApiClient : IDisposable
    {{
        private readonly HttpClient _httpClient;
        private readonly ILogger<{entidad}ApiClient> _logger;
        private readonly SemaphoreSlim _semaforo = new(3);

        public {entidad}ApiClient(
            HttpClient httpClient,
            ILogger<{entidad}ApiClient> logger)
        {{
            _httpClient = httpClient;
            _logger = logger;
        }}

        /// <summary>
        /// Obtiene un {entidad} con timeout y reintentos.
        /// CancellationToken permite cancelar operaciones largas.
        /// </summary>
        public async Task<{entidad}Dto?> ObtenerAsync(
            int id,
            CancellationToken cancellationToken = default)
        {{
            const int maxReintentos = 3;

            for (int intento = 1; intento <= maxReintentos; intento++)
            {{
                try
                {{
                    var response = await _httpClient.GetAsync(
                        $"api/{entidad.lower()}s/{{id}}",
                        cancellationToken);

                    response.EnsureSuccessStatusCode();

                    var json = await response.Content.ReadAsStringAsync(
                        cancellationToken);

                    return JsonSerializer.Deserialize<{entidad}Dto>(json);
                }}
                catch (HttpRequestException ex) when (intento < maxReintentos)
                {{
                    _logger.LogWarning(
                        ex,
                        "Intento {{Intento}}/{{Max}} fallido para {entidad} {{Id}}",
                        intento, maxReintentos, id);

                    // Espera exponencial entre reintentos
                    await Task.Delay(
                        TimeSpan.FromSeconds(Math.Pow(2, intento)),
                        cancellationToken);
                }}
            }}

            return null;
        }}

        /// <summary>
        /// Obtiene múltiples {entidad}s en paralelo con control de concurrencia.
        /// SemaphoreSlim limita las llamadas simultáneas para no saturar la API.
        /// </summary>
        public async Task<List<{entidad}Dto>> ObtenerVariosAsync(
            IEnumerable<int> ids,
            CancellationToken cancellationToken = default)
        {{
            var tareas = new List<Task<{entidad}Dto?>>();

            foreach (var id in ids)
            {{
                // El semáforo limita la concurrencia a 3 llamadas simultáneas
                await _semaforo.WaitAsync(cancellationToken);

                tareas.Add(Task.Run(async () =>
                {{
                    try
                    {{
                        return await ObtenerAsync(id, cancellationToken);
                    }}
                    finally
                    {{
                        _semaforo.Release();
                    }}
                }}, cancellationToken));
            }}

            // Task.WhenAll ejecuta todas las tareas en paralelo
            var resultados = await Task.WhenAll(tareas);

            return resultados
                .Where(r => r != null)
                .Select(r => r!)
                .ToList();
        }}

        /// <summary>
        /// Stream asíncrono con IAsyncEnumerable (C# 8+).
        /// Permite procesar items a medida que llegan, sin cargar
        /// todo en memoria. Ideal para datasets grandes.
        /// </summary>
        public async IAsyncEnumerable<{entidad}Dto> StreamAsync(
            [System.Runtime.CompilerServices.EnumeratorCancellation]
            CancellationToken cancellationToken = default)
        {{
            int pagina = 1;
            bool hayMas = true;

            while (hayMas && !cancellationToken.IsCancellationRequested)
            {{
                var response = await _httpClient.GetAsync(
                    $"api/{entidad.lower()}s?page={{pagina}}",
                    cancellationToken);

                var items = await JsonSerializer.DeserializeAsync<List<{entidad}Dto>>(
                    await response.Content.ReadAsStreamAsync(cancellationToken),
                    cancellationToken: cancellationToken);

                if (items == null || items.Count == 0)
                {{
                    hayMas = false;
                    yield break;
                }}

                foreach (var item in items)
                {{
                    yield return item;
                }}

                pagina++;
            }}
        }}

        public void Dispose()
        {{
            _semaforo?.Dispose();
            _httpClient?.Dispose();
        }}
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="async",
            patron="Async Client",
            complejidad=4,
            descripcion=f"Cliente HTTP asíncrono para {entidad} con reintentos y concurrencia",
            palabras_clave=[entidad.lower(), "async", "await", "http", "paralelo", "task", "concurrencia"],
            usa_async=True,
            usa_generics=True
        )

    # ----------------------------------------------------------------
    # 2.7: Generación de Middleware y Validadores
    # ----------------------------------------------------------------

    def generar_middleware(self) -> SnippetCSharp:
        """
        Genera middleware de ASP.NET Core para manejo global de errores.

        NOTA EDUCATIVA:
            El middleware es un patrón de "pipeline" donde cada componente
            procesa la request y puede decidir pasar al siguiente o
            cortocircuitar la cadena. Los generadores de código aprenden
            este patrón porque es muy repetitivo en proyectos ASP.NET Core.
        """
        codigo = """using System;
using System.Net;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace MiApp.Api.Middleware
{
    /// <summary>
    /// Middleware global para manejo de excepciones.
    /// Intercepta todas las excepciones no manejadas y retorna
    /// respuestas HTTP apropiadas con formato consistente.
    /// </summary>
    public class GlobalExceptionMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ILogger<GlobalExceptionMiddleware> _logger;

        public GlobalExceptionMiddleware(
            RequestDelegate next,
            ILogger<GlobalExceptionMiddleware> logger)
        {
            _next = next;
            _logger = logger;
        }

        /// <summary>
        /// Método principal del middleware.
        /// Envuelve toda la pipeline en un try-catch global.
        /// </summary>
        public async Task InvokeAsync(HttpContext context)
        {
            try
            {
                // Pasar al siguiente middleware en la pipeline
                await _next(context);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error no manejado: {Message}", ex.Message);
                await ManejarExcepcionAsync(context, ex);
            }
        }

        /// <summary>
        /// Convierte excepciones en respuestas HTTP estructuradas.
        /// Mapea tipos de excepción a códigos HTTP apropiados.
        /// </summary>
        private static async Task ManejarExcepcionAsync(
            HttpContext context, Exception excepcion)
        {
            var (statusCode, mensaje) = excepcion switch
            {
                ArgumentNullException => (HttpStatusCode.BadRequest,
                    "Parámetro requerido no proporcionado"),
                KeyNotFoundException => (HttpStatusCode.NotFound,
                    "Recurso no encontrado"),
                UnauthorizedAccessException => (HttpStatusCode.Unauthorized,
                    "No autorizado"),
                InvalidOperationException => (HttpStatusCode.Conflict,
                    "Operación no válida en el estado actual"),
                _ => (HttpStatusCode.InternalServerError,
                    "Error interno del servidor")
            };

            context.Response.ContentType = "application/json";
            context.Response.StatusCode = (int)statusCode;

            var respuesta = new
            {
                error = true,
                message = mensaje,
                statusCode = (int)statusCode,
                timestamp = DateTime.UtcNow
            };

            var json = JsonSerializer.Serialize(respuesta, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            await context.Response.WriteAsync(json);
        }
    }

    /// <summary>
    /// Extensión para registrar el middleware de forma limpia.
    /// Sigue la convención Use{Nombre} de ASP.NET Core.
    /// </summary>
    public static class GlobalExceptionMiddlewareExtensions
    {
        public static IApplicationBuilder UseGlobalExceptionHandler(
            this IApplicationBuilder app)
        {
            return app.UseMiddleware<GlobalExceptionMiddleware>();
        }
    }
}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="middleware",
            patron="Middleware Pipeline",
            complejidad=3,
            descripcion="Middleware global de manejo de excepciones para ASP.NET Core",
            palabras_clave=["middleware", "exception", "error", "pipeline", "http", "global"],
            usa_async=True
        )

    # ----------------------------------------------------------------
    # 2.8: Generación de Validadores con FluentValidation
    # ----------------------------------------------------------------

    def generar_validador(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera un validador usando el patrón FluentValidation.
        """
        entidad = entidad or random.choice(self.ENTIDADES)

        codigo = f"""using FluentValidation;
using MiApp.Core.DTOs;

namespace MiApp.Core.Validators
{{
    /// <summary>
    /// Validador para Create{entidad}Dto usando FluentValidation.
    /// Las reglas de validación se definen de forma declarativa
    /// y se ejecutan automáticamente en la pipeline de ASP.NET Core.
    /// </summary>
    public class Create{entidad}DtoValidator : AbstractValidator<Create{entidad}Dto>
    {{
        public Create{entidad}DtoValidator()
        {{
            RuleFor(x => x.Nombre)
                .NotEmpty().WithMessage("El nombre es obligatorio")
                .MaximumLength(200).WithMessage("El nombre no puede exceder 200 caracteres")
                .Matches(@"^[a-zA-ZáéíóúÁÉÍÓÚñÑ\\s]+$")
                .WithMessage("El nombre solo puede contener letras");

            RuleFor(x => x.Descripcion)
                .MaximumLength(1000)
                .WithMessage("La descripción no puede exceder 1000 caracteres");

            RuleFor(x => x.Precio)
                .GreaterThan(0).WithMessage("El precio debe ser mayor a cero")
                .LessThan(1000000).WithMessage("El precio no puede exceder 1,000,000");

            RuleFor(x => x.Email)
                .EmailAddress().When(x => !string.IsNullOrEmpty(x.Email))
                .WithMessage("El email no tiene un formato válido");

            RuleFor(x => x.CategoriaId)
                .GreaterThan(0).WithMessage("Debe seleccionar una categoría válida");
        }}
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="validador",
            patron="FluentValidation",
            complejidad=2,
            descripcion=f"Validador de reglas de negocio para {entidad}",
            palabras_clave=[entidad.lower(), "validacion", "reglas", "fluent", "validator"],
            usa_generics=True
        )

    # ----------------------------------------------------------------
    # 2.9: Generación de patrones de Microservicios
    # ----------------------------------------------------------------

    def generar_event_handler(self, entidad: Optional[str] = None) -> SnippetCSharp:
        """
        Genera un manejador de eventos para comunicación entre microservicios.
        """
        entidad = entidad or random.choice(self.ENTIDADES)

        codigo = f"""using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MediatR;
using Microsoft.Extensions.Logging;

namespace MiApp.Core.Events
{{
    /// <summary>
    /// Evento de dominio que se dispara cuando se crea un {entidad}.
    /// Los eventos permiten comunicación desacoplada entre componentes.
    /// En microservicios, estos eventos pueden publicarse a un bus
    /// de mensajes (RabbitMQ, Azure Service Bus, etc.).
    /// </summary>
    public record {entidad}CreadoEvent : INotification
    {{
        public int {entidad}Id {{ get; init; }}
        public string Nombre {{ get; init; }} = string.Empty;
        public DateTime FechaCreacion {{ get; init; }}
        public string CreadoPor {{ get; init; }} = string.Empty;
    }}

    /// <summary>
    /// Handler que procesa el evento de creación de {entidad}.
    /// Puede enviar notificaciones, actualizar caché, sincronizar
    /// con otros servicios, etc.
    /// </summary>
    public class {entidad}CreadoEventHandler
        : INotificationHandler<{entidad}CreadoEvent>
    {{
        private readonly ILogger<{entidad}CreadoEventHandler> _logger;
        private readonly INotificationService _notificationService;

        public {entidad}CreadoEventHandler(
            ILogger<{entidad}CreadoEventHandler> logger,
            INotificationService notificationService)
        {{
            _logger = logger;
            _notificationService = notificationService;
        }}

        public async Task Handle(
            {entidad}CreadoEvent evento,
            CancellationToken cancellationToken)
        {{
            _logger.LogInformation(
                "Procesando evento: {entidad} {{Id}} creado por {{User}}",
                evento.{entidad}Id, evento.CreadoPor);

            // Enviar notificación a usuarios suscritos
            await _notificationService.EnviarAsync(
                $"Nuevo {entidad}: {{evento.Nombre}}",
                cancellationToken);

            // Publicar evento a bus de mensajes para otros microservicios
            _logger.LogInformation(
                "Evento {entidad}Creado publicado exitosamente");
        }}
    }}
}}
"""
        return SnippetCSharp(
            codigo=codigo,
            categoria="evento",
            patron="Event-Driven / MediatR",
            complejidad=4,
            descripcion=f"Evento de dominio y handler para {entidad} (patrón microservicios)",
            palabras_clave=[entidad.lower(), "evento", "mediator", "microservicio", "handler", "notificacion"],
            usa_async=True,
            usa_generics=True
        )

    # ----------------------------------------------------------------
    # Helpers internos
    # ----------------------------------------------------------------

    def _generar_propiedades(self, entidad: str, cantidad: int) -> List[Tuple[str, str, str]]:
        """Genera propiedades realistas para una entidad C#."""
        propiedades_comunes = [
            ("Id", "int", f"Identificador único del {entidad}"),
            ("Nombre", "string", f"Nombre del {entidad}"),
            ("Descripcion", "string?", f"Descripción detallada del {entidad}"),
            ("Precio", "decimal", "Precio en la moneda local"),
            ("Email", "string", "Dirección de correo electrónico"),
            ("Telefono", "string?", "Número de teléfono de contacto"),
            ("FechaCreacion", "DateTime", "Fecha y hora de creación del registro"),
            ("FechaModificacion", "DateTime?", "Última fecha de modificación"),
            ("EstaActivo", "bool", f"Indica si el {entidad} está activo"),
            ("CategoriaId", "int", "ID de la categoría asociada"),
            ("Cantidad", "int", "Cantidad disponible"),
            ("Codigo", "string", "Código único de referencia"),
        ]
        return propiedades_comunes[:cantidad]

    def _filtrar_props_creacion(self, propiedades: List[Tuple[str, str, str]]) -> str:
        """Filtra propiedades para DTO de creación (sin Id ni fechas)."""
        excluir = {"Id", "FechaCreacion", "FechaModificacion"}
        resultado = ""
        for nombre, tipo, desc in propiedades:
            if nombre not in excluir:
                resultado += f"\n        public {tipo} {nombre} {{ get; set; }}"
        return resultado


# ============================================================================
# SECCIÓN 3: Clase principal del Dataset
# ============================================================================

class DatasetCSharp:
    """
    Genera y gestiona el dataset completo de código C# para entrenamiento.

    NOTA EDUCATIVA — COMPOSICIÓN DEL DATASET:
        Un buen dataset de código debe tener:
        1. VARIEDAD: diferentes patrones (Controller, Service, DTO, etc.)
        2. BALANCE: no 90% DTOs simples y 10% código complejo
        3. COHERENCIA: las entidades deben reutilizarse (Product aparece
           tanto en DTO como en Controller y Service)
        4. PROGRESIÓN: desde código simple hasta patrones avanzados

        Los datasets reales (como The Stack de HuggingFace) tienen
        millones de archivos. Nuestro dataset sintético es pequeño
        pero cubre los patrones más importantes.
    """

    def __init__(self, semilla: int = 42):
        self.generador = GeneradorPatronesCSharp(semilla)
        self.snippets: List[SnippetCSharp] = []

    def generar_dataset(self, tamano: int = 100) -> List[SnippetCSharp]:
        """
        Genera el dataset completo balanceado por categorías.

        Distribución objetivo (similar a proyectos reales):
        - 20% DTOs (son los más comunes en proyectos reales)
        - 15% Interfaces
        - 20% Services
        - 15% Controllers
        - 10% LINQ
        - 10% Async
        - 5% Middleware
        - 5% Eventos
        """
        distribuciones = {
            'dto': int(tamano * 0.20),
            'interfaz': int(tamano * 0.15),
            'servicio': int(tamano * 0.20),
            'controller': int(tamano * 0.15),
            'linq': int(tamano * 0.10),
            'async': int(tamano * 0.10),
            'middleware': max(1, int(tamano * 0.05)),
            'evento': max(1, int(tamano * 0.05)),
            'validador': max(1, int(tamano * 0.05)),
        }

        self.snippets = []

        # Generar snippets coherentes: misma entidad en diferentes capas
        for entidad in self.generador.ENTIDADES:
            self.snippets.append(self.generador.generar_dto(entidad))
            self.snippets.append(self.generador.generar_interfaz_servicio(entidad))
            self.snippets.append(self.generador.generar_servicio(entidad))
            self.snippets.append(self.generador.generar_controller(entidad))

        # Rellenar con snippets adicionales hasta alcanzar el tamaño
        generadores = [
            self.generador.generar_linq_avanzado,
            self.generador.generar_patron_async,
            self.generador.generar_validador,
            self.generador.generar_event_handler,
        ]

        while len(self.snippets) < tamano:
            gen = random.choice(generadores)
            self.snippets.append(gen())

        # Agregar middlewares
        for _ in range(distribuciones.get('middleware', 1)):
            self.snippets.append(self.generador.generar_middleware())

        random.shuffle(self.snippets)
        return self.snippets

    def obtener_textos_entrenamiento(self) -> List[str]:
        """Extrae solo el código de los snippets para entrenamiento."""
        return [s.codigo for s in self.snippets]

    def obtener_estadisticas(self) -> Dict:
        """Calcula estadísticas del dataset generado."""
        stats = {
            "total_snippets": len(self.snippets),
            "por_categoria": {},
            "complejidad_promedio": 0,
            "con_linq": 0,
            "con_async": 0,
            "con_generics": 0,
            "total_lineas": 0,
            "total_caracteres": 0,
        }

        for s in self.snippets:
            cat = s.categoria
            stats["por_categoria"][cat] = stats["por_categoria"].get(cat, 0) + 1
            stats["complejidad_promedio"] += s.complejidad
            stats["con_linq"] += int(s.usa_linq)
            stats["con_async"] += int(s.usa_async)
            stats["con_generics"] += int(s.usa_generics)
            stats["total_lineas"] += s.codigo.count('\n')
            stats["total_caracteres"] += len(s.codigo)

        if self.snippets:
            stats["complejidad_promedio"] /= len(self.snippets)

        return stats


def generar_dataset_completo(tamano: int = 100, semilla: int = 42) -> DatasetCSharp:
    """
    Función de conveniencia para generar el dataset completo.
    """
    dataset = DatasetCSharp(semilla=semilla)
    dataset.generar_dataset(tamano)
    return dataset
