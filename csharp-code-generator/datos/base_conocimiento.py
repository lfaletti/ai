"""
============================================================================
datos/base_conocimiento.py — Base de Conocimiento para el Sistema RAG
============================================================================

PROPÓSITO EDUCATIVO:
    Este módulo crea la BASE DE CONOCIMIENTO que alimenta el sistema RAG.
    Contiene snippets de código C# organizados por categoría, con metadatos
    que permiten búsqueda semántica eficiente.

¿QUÉ ES UNA BASE DE CONOCIMIENTO EN RAG?
    Es una colección de documentos (en nuestro caso, snippets de código)
    que el modelo puede CONSULTAR antes de generar código nuevo.

    Analogía: Es como un desarrollador que, antes de escribir un nuevo
    Controller, revisa los controllers existentes del proyecto para
    mantener consistencia.

¿POR QUÉ RAG ES ESPECIALMENTE ÚTIL PARA CÓDIGO?
    1. CONSISTENCIA: El código generado sigue los patrones del proyecto
    2. PRECISIÓN: Usa las mismas librerías y convenciones existentes
    3. CONTEXTO: Entiende la arquitectura del proyecto
    4. REDUCCIÓN DE ALUCINACIONES: No inventa APIs que no existen
============================================================================
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json


@dataclass
class DocumentoRAG:
    """
    Un documento en la base de conocimiento para RAG.

    NOTA EDUCATIVA:
        Cada documento tiene:
        - contenido: el código C# en sí (se convierte en embedding)
        - metadatos: información estructurada para filtrar búsquedas
        - descripcion_natural: texto en español que describe el código
          (esto es CRUCIAL porque la query del usuario es en lenguaje natural,
           y necesitamos hacer match entre "crear endpoint REST" y el código
           de un Controller)
    """
    id: str                             # Identificador único
    contenido: str                      # Código C#
    descripcion_natural: str            # Descripción en español
    categoria: str                      # "controller", "service", etc.
    etiquetas: List[str] = field(default_factory=list)
    entidad: str = ""                   # Entidad principal (Product, User, etc.)
    patron: str = ""                    # Patrón arquitectónico
    complejidad: int = 1


class BaseConocimientoRAG:
    """
    Base de conocimiento que almacena snippets de código C# para RAG.

    FLUJO RAG COMPLETO:
    1. Desarrollador escribe: "necesito un endpoint para crear productos"
    2. Se convierte a embedding (vector numérico)
    3. Se buscan los documentos más similares en FAISS
    4. Se recuperan los top-K snippets relevantes
    5. Se concatenan como contexto antes del prompt
    6. El modelo genera código INFORMADO por el contexto

    DIFERENCIA CON GENERACIÓN SIN RAG:
    - Sin RAG: el modelo genera "de memoria" → puede no seguir convenciones
    - Con RAG: el modelo tiene ejemplos del proyecto → genera código consistente
    """

    def __init__(self):
        self.documentos: List[DocumentoRAG] = []
        self._indice_por_categoria: Dict[str, List[int]] = {}
        self._indice_por_etiqueta: Dict[str, List[int]] = {}

    def agregar_documento(self, doc: DocumentoRAG):
        """
        Agrega un documento a la base de conocimiento.

        Se mantienen índices invertidos para búsqueda rápida por categoría
        y etiquetas. En producción, estos índices serían reemplazados por
        FAISS para búsqueda por similitud semántica.
        """
        idx = len(self.documentos)
        self.documentos.append(doc)

        # Índice por categoría
        if doc.categoria not in self._indice_por_categoria:
            self._indice_por_categoria[doc.categoria] = []
        self._indice_por_categoria[doc.categoria].append(idx)

        # Índice por etiquetas
        for etiqueta in doc.etiquetas:
            if etiqueta not in self._indice_por_etiqueta:
                self._indice_por_etiqueta[etiqueta] = []
            self._indice_por_etiqueta[etiqueta].append(idx)

    def buscar_por_categoria(self, categoria: str) -> List[DocumentoRAG]:
        """Búsqueda exacta por categoría (para comparar con búsqueda semántica)."""
        indices = self._indice_por_categoria.get(categoria, [])
        return [self.documentos[i] for i in indices]

    def buscar_por_etiquetas(self, etiquetas: List[str]) -> List[DocumentoRAG]:
        """Búsqueda por etiquetas con ranking por relevancia."""
        puntuaciones: Dict[int, int] = {}

        for etiqueta in etiquetas:
            for idx in self._indice_por_etiqueta.get(etiqueta, []):
                puntuaciones[idx] = puntuaciones.get(idx, 0) + 1

        # Ordenar por puntuación descendente
        indices_ordenados = sorted(
            puntuaciones.keys(),
            key=lambda i: puntuaciones[i],
            reverse=True
        )

        return [self.documentos[i] for i in indices_ordenados]

    def obtener_todos(self) -> List[DocumentoRAG]:
        """Retorna todos los documentos."""
        return self.documentos

    def obtener_textos(self) -> List[str]:
        """
        Retorna textos para crear embeddings.

        NOTA EDUCATIVA:
            Combinamos descripción natural + código porque queremos
            que la búsqueda semántica encuentre coincidencias tanto
            por la descripción ("crear endpoint REST") como por
            el código en sí ("[HttpPost]", "[ApiController]").
        """
        return [
            f"{doc.descripcion_natural}\n\n{doc.contenido}"
            for doc in self.documentos
        ]

    def obtener_estadisticas(self) -> Dict:
        """Estadísticas de la base de conocimiento."""
        return {
            "total_documentos": len(self.documentos),
            "categorias": {k: len(v) for k, v in self._indice_por_categoria.items()},
            "etiquetas_unicas": len(self._indice_por_etiqueta),
            "total_caracteres": sum(len(d.contenido) for d in self.documentos),
        }


# ============================================================================
# Snippets adicionales específicos para la base de conocimiento RAG
# ============================================================================

SNIPPETS_BASE_CONOCIMIENTO = [
    {
        "id": "rag_controller_product",
        "descripcion": "Controller REST completo para gestionar productos con operaciones CRUD, paginación y búsqueda",
        "categoria": "controller",
        "etiquetas": ["controller", "rest", "api", "crud", "producto", "endpoint", "http"],
        "entidad": "Product",
        "patron": "REST Controller",
        "codigo": """[ApiController]
[Route("api/[controller]")]
public class ProductsController : ControllerBase
{
    private readonly IProductService _service;

    public ProductsController(IProductService service)
    {
        _service = service;
    }

    [HttpGet]
    public async Task<ActionResult<PagedResult<ProductDto>>> GetAll(
        [FromQuery] int page = 1, [FromQuery] int pageSize = 10)
    {
        var result = await _service.GetAllAsync(page, pageSize);
        return Ok(result);
    }

    [HttpGet("{id}")]
    public async Task<ActionResult<ProductDto>> GetById(int id)
    {
        var product = await _service.GetByIdAsync(id);
        if (product == null) return NotFound();
        return Ok(product);
    }

    [HttpPost]
    public async Task<ActionResult<ProductDto>> Create([FromBody] CreateProductDto dto)
    {
        var created = await _service.CreateAsync(dto);
        return CreatedAtAction(nameof(GetById), new { id = created.Id }, created);
    }

    [HttpPut("{id}")]
    public async Task<ActionResult<ProductDto>> Update(int id, [FromBody] UpdateProductDto dto)
    {
        var updated = await _service.UpdateAsync(id, dto);
        return Ok(updated);
    }

    [HttpDelete("{id}")]
    public async Task<IActionResult> Delete(int id)
    {
        await _service.DeleteAsync(id);
        return NoContent();
    }
}"""
    },
    {
        "id": "rag_service_order",
        "descripcion": "Servicio de pedidos con lógica de negocio, validación y cálculo de totales usando LINQ",
        "categoria": "servicio",
        "etiquetas": ["servicio", "negocio", "pedido", "orden", "calculo", "linq", "validacion"],
        "entidad": "Order",
        "patron": "Business Service",
        "codigo": """public class OrderService : IOrderService
{
    private readonly IOrderRepository _repository;
    private readonly IProductService _productService;
    private readonly ILogger<OrderService> _logger;

    public OrderService(
        IOrderRepository repository,
        IProductService productService,
        ILogger<OrderService> logger)
    {
        _repository = repository;
        _productService = productService;
        _logger = logger;
    }

    public async Task<OrderDto> CreateOrderAsync(CreateOrderDto dto)
    {
        // Validar que todos los productos existen
        var productIds = dto.Items.Select(i => i.ProductId).Distinct();
        var products = await _productService.GetByIdsAsync(productIds);

        if (products.Count() != productIds.Count())
            throw new ValidationException("Algunos productos no existen");

        // Calcular totales usando LINQ
        var orderItems = dto.Items.Select(item =>
        {
            var product = products.First(p => p.Id == item.ProductId);
            return new OrderItem
            {
                ProductId = item.ProductId,
                Quantity = item.Quantity,
                UnitPrice = product.Price,
                Subtotal = product.Price * item.Quantity
            };
        }).ToList();

        var order = new Order
        {
            CustomerId = dto.CustomerId,
            Items = orderItems,
            Total = orderItems.Sum(i => i.Subtotal),
            CreatedAt = DateTime.UtcNow,
            Status = OrderStatus.Pending
        };

        await _repository.AddAsync(order);
        await _repository.SaveChangesAsync();

        return _mapper.Map<OrderDto>(order);
    }
}"""
    },
    {
        "id": "rag_repository_generic",
        "descripcion": "Repositorio genérico base con Entity Framework Core para operaciones CRUD reutilizables",
        "categoria": "repositorio",
        "etiquetas": ["repositorio", "entity framework", "generico", "crud", "base", "datos", "ef core"],
        "entidad": "Generic",
        "patron": "Generic Repository",
        "codigo": """public class Repository<T> : IRepository<T> where T : class, IEntity
{
    protected readonly DbContext _context;
    protected readonly DbSet<T> _dbSet;

    public Repository(DbContext context)
    {
        _context = context;
        _dbSet = context.Set<T>();
    }

    public virtual async Task<T?> GetByIdAsync(int id)
    {
        return await _dbSet.FindAsync(id);
    }

    public virtual async Task<IEnumerable<T>> GetAllAsync()
    {
        return await _dbSet.ToListAsync();
    }

    public virtual async Task<IEnumerable<T>> FindAsync(
        Expression<Func<T, bool>> predicate)
    {
        return await _dbSet.Where(predicate).ToListAsync();
    }

    public virtual async Task<T> AddAsync(T entity)
    {
        var entry = await _dbSet.AddAsync(entity);
        return entry.Entity;
    }

    public virtual void Update(T entity)
    {
        _dbSet.Update(entity);
    }

    public virtual void Remove(T entity)
    {
        _dbSet.Remove(entity);
    }

    public async Task<int> SaveChangesAsync()
    {
        return await _context.SaveChangesAsync();
    }
}"""
    },
    {
        "id": "rag_dto_pagination",
        "descripcion": "DTO de resultado paginado genérico para APIs REST con información de paginación",
        "categoria": "dto",
        "etiquetas": ["dto", "paginacion", "generico", "api", "resultado", "lista"],
        "entidad": "Pagination",
        "patron": "Pagination DTO",
        "codigo": """public class PagedResult<T>
{
    public IReadOnlyList<T> Items { get; set; } = new List<T>();
    public int TotalCount { get; set; }
    public int PageNumber { get; set; }
    public int PageSize { get; set; }
    public int TotalPages => (int)Math.Ceiling(TotalCount / (double)PageSize);
    public bool HasPreviousPage => PageNumber > 1;
    public bool HasNextPage => PageNumber < TotalPages;

    public static PagedResult<T> Create(
        IReadOnlyList<T> items, int totalCount, int pageNumber, int pageSize)
    {
        return new PagedResult<T>
        {
            Items = items,
            TotalCount = totalCount,
            PageNumber = pageNumber,
            PageSize = pageSize
        };
    }
}"""
    },
    {
        "id": "rag_middleware_auth",
        "descripcion": "Middleware de autenticación JWT para APIs REST con validación de tokens",
        "categoria": "middleware",
        "etiquetas": ["middleware", "autenticacion", "jwt", "seguridad", "token", "auth"],
        "entidad": "Auth",
        "patron": "Authentication Middleware",
        "codigo": """public class JwtAuthenticationMiddleware
{
    private readonly RequestDelegate _next;
    private readonly IConfiguration _configuration;

    public JwtAuthenticationMiddleware(RequestDelegate next, IConfiguration config)
    {
        _next = next;
        _configuration = config;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var token = context.Request.Headers["Authorization"]
            .FirstOrDefault()?.Split(" ").Last();

        if (token != null)
        {
            try
            {
                var handler = new JwtSecurityTokenHandler();
                var key = Encoding.UTF8.GetBytes(_configuration["Jwt:Secret"]);

                handler.ValidateToken(token, new TokenValidationParameters
                {
                    ValidateIssuerSigningKey = true,
                    IssuerSigningKey = new SymmetricSecurityKey(key),
                    ValidateIssuer = true,
                    ValidIssuer = _configuration["Jwt:Issuer"],
                    ValidateAudience = false,
                    ClockSkew = TimeSpan.Zero
                }, out var validatedToken);

                var jwtToken = (JwtSecurityToken)validatedToken;
                var userId = jwtToken.Claims.First(c => c.Type == "sub").Value;

                context.Items["UserId"] = userId;
            }
            catch (Exception)
            {
                // Token inválido, continuar sin autenticación
            }
        }

        await _next(context);
    }
}"""
    },
    {
        "id": "rag_linq_reporting",
        "descripcion": "Consultas LINQ avanzadas para reportes con agrupación, proyección y agregaciones",
        "categoria": "linq",
        "etiquetas": ["linq", "reporte", "agrupacion", "estadistica", "consulta", "agregacion"],
        "entidad": "Report",
        "patron": "LINQ Reporting",
        "codigo": """public class ReportService : IReportService
{
    private readonly IOrderRepository _orderRepo;

    public async Task<SalesReportDto> GenerateSalesReport(
        DateTime from, DateTime to)
    {
        var orders = await _orderRepo.GetOrdersInPeriod(from, to);

        var report = new SalesReportDto
        {
            Period = $"{from:yyyy-MM-dd} a {to:yyyy-MM-dd}",
            TotalSales = orders.Sum(o => o.Total),
            OrderCount = orders.Count(),
            AverageOrderValue = orders.Average(o => o.Total),

            // Ventas por categoría con LINQ GroupBy
            SalesByCategory = orders
                .SelectMany(o => o.Items)
                .GroupBy(i => i.Product.Category.Name)
                .Select(g => new CategorySalesDto
                {
                    Category = g.Key,
                    Total = g.Sum(i => i.Subtotal),
                    Quantity = g.Sum(i => i.Quantity),
                    AveragePrice = g.Average(i => i.UnitPrice)
                })
                .OrderByDescending(c => c.Total)
                .ToList(),

            // Top productos más vendidos
            TopProducts = orders
                .SelectMany(o => o.Items)
                .GroupBy(i => i.ProductId)
                .Select(g => new TopProductDto
                {
                    ProductName = g.First().Product.Name,
                    TotalSold = g.Sum(i => i.Quantity),
                    Revenue = g.Sum(i => i.Subtotal)
                })
                .OrderByDescending(p => p.Revenue)
                .Take(10)
                .ToList()
        };

        return report;
    }
}"""
    },
    {
        "id": "rag_config_startup",
        "descripcion": "Configuración de startup de ASP.NET Core con inyección de dependencias y pipeline de middleware",
        "categoria": "configuracion",
        "etiquetas": ["startup", "configuracion", "di", "inyeccion", "dependencias", "pipeline", "program"],
        "entidad": "Startup",
        "patron": "ASP.NET Core Startup",
        "codigo": """var builder = WebApplication.CreateBuilder(args);

// Configurar servicios
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    });

// Registrar servicios de negocio (DI)
builder.Services.AddScoped<IProductService, ProductService>();
builder.Services.AddScoped<IOrderService, OrderService>();
builder.Services.AddScoped(typeof(IRepository<>), typeof(Repository<>));

// Configurar Entity Framework
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("Default")));

// Configurar AutoMapper
builder.Services.AddAutoMapper(typeof(MappingProfile));

// Configurar FluentValidation
builder.Services.AddValidatorsFromAssemblyContaining<CreateProductDtoValidator>();

// Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Pipeline de middleware
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseGlobalExceptionHandler();
app.UseHttpsRedirection();
app.UseAuthentication();
app.UseAuthorization();
app.MapControllers();

app.Run();"""
    },
]


def construir_base_conocimiento() -> BaseConocimientoRAG:
    """
    Construye la base de conocimiento completa para RAG.

    Combina:
    1. Los snippets predefinidos (base_conocimiento)
    2. Snippets generados dinámicamente del dataset

    NOTA EDUCATIVA:
        En producción, la base de conocimiento se construiría
        indexando todo el código fuente del proyecto. Aquí usamos
        snippets predefinidos para tener control total sobre
        qué ejemplos están disponibles para el RAG.
    """
    base = BaseConocimientoRAG()

    for snippet_data in SNIPPETS_BASE_CONOCIMIENTO:
        doc = DocumentoRAG(
            id=snippet_data["id"],
            contenido=snippet_data["codigo"],
            descripcion_natural=snippet_data["descripcion"],
            categoria=snippet_data["categoria"],
            etiquetas=snippet_data["etiquetas"],
            entidad=snippet_data.get("entidad", ""),
            patron=snippet_data.get("patron", ""),
        )
        base.agregar_documento(doc)

    return base
