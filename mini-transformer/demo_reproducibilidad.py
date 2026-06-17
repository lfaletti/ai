"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              DEMO DE REPRODUCIBILIDAD EN MINI TRANSFORMER                  ║
║                                                                            ║
║  Este script demuestra por qué, sin fijar semillas (seeds), cada           ║
║  ejecución de entrenamiento produce un modelo DIFERENTE, y cómo            ║
║  fijar las semillas garantiza resultados IDÉNTICOS.                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

FUENTES DE ALEATORIEDAD EN EL ENTRENAMIENTO:
=============================================

1. INICIALIZACIÓN DE PESOS (nn.init.xavier_uniform_, nn.Embedding, etc.)
   → Los pesos iniciales del modelo son aleatorios.
   → Sin semilla fija, cada modelo empieza en un punto diferente del
     espacio de parámetros.

2. DATALOADER con shuffle=True
   → El orden en que el modelo ve los datos cambia cada época.
   → Diferentes órdenes producen diferentes actualizaciones de gradientes.

3. DROPOUT (nn.Dropout con p=0.1)
   → Durante entrenamiento, dropout desactiva neuronas al azar.
   → Actúa como regularización, pero introduce aleatoriedad.
   → Las neuronas desactivadas son diferentes en cada ejecución.

4. GENERACIÓN DE TEXTO (torch.multinomial)
   → El sampling de tokens es probabilístico.
   → Incluso con el mismo modelo, generar texto produce resultados
     distintos cada vez (a menos que se fije la semilla).

CÓMO CONTROLAR LA ALEATORIEDAD:
================================

Para obtener resultados reproducibles necesitamos fijar TODAS las fuentes:

    torch.manual_seed(42)           # Generador de PyTorch (CPU)
    torch.cuda.manual_seed_all(42)  # Generador de PyTorch (todas las GPUs)
    random.seed(42)                 # Módulo random de Python
    np.random.seed(42)              # NumPy (si se usa)

    # Y además, para operaciones deterministas en GPU:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader necesita un worker_init_fn y un generator con semilla:
    g = torch.Generator()
    g.manual_seed(42)
    DataLoader(..., generator=g, worker_init_fn=seed_worker)
"""

import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from mini_transformer import MiniTransformer, CharTokenizer, TextDataset


# =============================================================================
# TEXTO DE ENTRENAMIENTO (corto para que la demo sea rápida)
# =============================================================================
DEMO_TEXT = """
El dragón leía libros bajo las estrellas brillantes.
Cada noche subía a la cima de la montaña para contar constelaciones.
Una exploradora valiente llegó buscando aventuras y conocimiento.
Juntos descubrieron que la curiosidad transforma el mundo.
Los niños de la aldea subían a escuchar historias maravillosas.
El pequeño dragón sabio se convirtió en el maestro más querido.
"""

# =============================================================================
# CONFIGURACIÓN REDUCIDA (para que cada entrenamiento sea rápido)
# =============================================================================
TRAIN_CONFIG = {
    'd_model': 32,
    'num_heads': 2,
    'num_layers': 1,
    'd_ff': 64,
    'seq_len': 16,
    'batch_size': 8,
    'num_epochs': 5,        # pocas épocas para demo rápida
    'learning_rate': 3e-4,
}


# =============================================================================
# FUNCIÓN AUXILIAR: Fijar TODAS las semillas
# =============================================================================
def fijar_semillas(seed: int = 42):
    """
    Fija todas las fuentes de aleatoriedad para garantizar reproducibilidad.

    Args:
        seed: Valor de la semilla (cualquier entero, 42 es tradición 🙂)

    ¿POR QUÉ TANTAS LÍNEAS?
    Cada librería/componente tiene su propio generador de números aleatorios.
    Si olvidamos fijar alguno, la reproducibilidad se rompe.
    """
    # 1. Módulo random de Python (usado internamente por algunos componentes)
    random.seed(seed)

    # 2. NumPy (usado por muchas librerías de datos)
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch GPU (todas las GPUs disponibles)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Para multi-GPU

    # 5. Operaciones deterministas en cuDNN
    #    - deterministic=True: fuerza algoritmos deterministas (puede ser más lento)
    #    - benchmark=False: desactiva la búsqueda automática de algoritmos rápidos
    #      (la búsqueda es no-determinista)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. Variables de entorno para operaciones atómicas deterministas
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    """
    Función para inicializar la semilla en cada worker del DataLoader.

    Cuando num_workers > 0, cada worker es un proceso separado con su
    propio estado aleatorio. Sin esta función, los workers podrían
    generar datos en orden diferente entre ejecuciones.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# FUNCIÓN DE ENTRENAMIENTO (simplificada para la demo)
# =============================================================================
def entrenar_mini(config: dict, tokenizer: CharTokenizer, text: str,
                  seed: int = None, verbose: bool = True) -> dict:
    """
    Entrena un MiniTransformer y retorna información del resultado.

    Args:
        config: Diccionario de configuración
        tokenizer: Tokenizer ya ajustado
        text: Texto de entrenamiento
        seed: Si se proporciona, fija todas las semillas antes de entrenar
        verbose: Si imprimir progreso

    Returns:
        Dict con los pesos finales y la pérdida
    """
    # ─── Opcionalmente fijar semillas ───
    if seed is not None:
        fijar_semillas(seed)
        if verbose:
            print(f"  🌱 Semilla fijada: {seed}")
    else:
        if verbose:
            print(f"  🎲 Sin semilla (aleatorio)")

    # ─── Crear dataset y dataloader ───
    dataset = TextDataset(text, tokenizer, seq_len=config['seq_len'])

    # NOTA: Para reproducibilidad total del DataLoader con shuffle=True,
    # necesitamos pasar un Generator con semilla fija.
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True,
            generator=g,                # ← Generator con semilla fija
            worker_init_fn=seed_worker,  # ← Semilla por worker
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True,
        )

    # ─── Crear modelo ───
    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['seq_len'] * 2,
        dropout=0.1,
        token_to_id=tokenizer.token_to_id,
        id_to_token=tokenizer.id_to_token,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # ─── Entrenar ───
    model.train()
    final_loss = 0
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        n = 0
        for inputs, targets in dataloader:
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        final_loss = epoch_loss / max(n, 1)
        if verbose:
            print(f"    Época {epoch+1}/{config['num_epochs']}: loss={final_loss:.4f}")

    # ─── Extraer los primeros pesos para comparar ───
    # Tomamos los primeros parámetros del embedding como "huella digital"
    embedding_weights = model.token_embedding.weight.data.clone()
    first_layer_attn = model.layers[0].self_attention.W_q.weight.data.clone()

    return {
        'final_loss': final_loss,
        'embedding_weights': embedding_weights,
        'attn_weights': first_layer_attn,
        'model': model,
    }


def comparar_pesos(result_a: dict, result_b: dict, nombre_a: str, nombre_b: str) -> bool:
    """
    Compara los pesos de dos entrenamientos y reporta si son idénticos.

    Returns:
        True si los pesos son idénticos, False si son diferentes.
    """
    # Comparar embedding weights
    emb_iguales = torch.equal(result_a['embedding_weights'], result_b['embedding_weights'])

    # Comparar pesos de atención
    attn_iguales = torch.equal(result_a['attn_weights'], result_b['attn_weights'])

    # Diferencia numérica (para mostrar cuánto difieren)
    emb_diff = (result_a['embedding_weights'] - result_b['embedding_weights']).abs().mean().item()
    attn_diff = (result_a['attn_weights'] - result_b['attn_weights']).abs().mean().item()

    son_identicos = emb_iguales and attn_iguales

    print(f"\n  {'─' * 55}")
    print(f"  Comparación: {nombre_a} vs {nombre_b}")
    print(f"  {'─' * 55}")
    print(f"  Token Embedding iguales:  {'✅ SÍ' if emb_iguales else '❌ NO'}"
          f"  (diff media: {emb_diff:.8f})")
    print(f"  Attention W_q iguales:    {'✅ SÍ' if attn_iguales else '❌ NO'}"
          f"  (diff media: {attn_diff:.8f})")
    print(f"  Loss final A: {result_a['final_loss']:.6f}")
    print(f"  Loss final B: {result_b['final_loss']:.6f}")
    print(f"  Loss idéntico: {'✅ SÍ' if result_a['final_loss'] == result_b['final_loss'] else '❌ NO'}")
    print(f"\n  {'🟢 RESULTADO: Los modelos son IDÉNTICOS' if son_identicos else '🔴 RESULTADO: Los modelos son DIFERENTES'}")

    return son_identicos


# =============================================================================
# DEMO: Generación de texto con y sin semilla
# =============================================================================
def demo_generacion(model: MiniTransformer, tokenizer: CharTokenizer,
                    prompt: str, seed: int = None):
    """
    Genera texto y muestra que el sampling también es aleatorio.
    """
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long
    )
    if prompt_ids.size(1) == 0:
        print("  ⚠️  Prompt no reconocido por el tokenizer.")
        return ""

    if seed is not None:
        fijar_semillas(seed)

    model.eval()
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
    return tokenizer.decode(generated[0].tolist())


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
def main():
    print("╔" + "═" * 62 + "╗")
    print("║     DEMO DE REPRODUCIBILIDAD EN MINI TRANSFORMER              ║")
    print("╚" + "═" * 62 + "╝")

    # ─── Preparar tokenizer (esto es determinista, no depende de semilla) ───
    tokenizer = CharTokenizer()
    tokenizer.fit(DEMO_TEXT)

    # =====================================================================
    # PARTE 1: Entrenamiento SIN semilla fija → resultados DIFERENTES
    # =====================================================================
    print("\n" + "=" * 64)
    print("  PARTE 1: Entrenando 2 veces SIN fijar semilla")
    print("  Esperamos resultados DIFERENTES")
    print("=" * 64)

    print("\n  ▶ Entrenamiento A (sin semilla):")
    result_a = entrenar_mini(TRAIN_CONFIG, tokenizer, DEMO_TEXT, seed=None)

    print("\n  ▶ Entrenamiento B (sin semilla):")
    result_b = entrenar_mini(TRAIN_CONFIG, tokenizer, DEMO_TEXT, seed=None)

    diferentes = not comparar_pesos(result_a, result_b, "A (sin seed)", "B (sin seed)")

    if diferentes:
        print("\n  💡 EXPLICACIÓN: Sin semilla fija, cada entrenamiento usa")
        print("     números aleatorios diferentes para:")
        print("     • Inicialización de pesos (xavier_uniform_)")
        print("     • Orden de los datos (DataLoader shuffle=True)")
        print("     • Neuronas desactivadas por Dropout (p=0.1)")
        print("     → Por eso los modelos finales son DISTINTOS.")

    # =====================================================================
    # PARTE 2: Entrenamiento CON semilla fija → resultados IDÉNTICOS
    # =====================================================================
    print("\n\n" + "=" * 64)
    print("  PARTE 2: Entrenando 2 veces CON semilla fija (seed=42)")
    print("  Esperamos resultados IDÉNTICOS")
    print("=" * 64)

    SEED = 42

    print(f"\n  ▶ Entrenamiento C (seed={SEED}):")
    result_c = entrenar_mini(TRAIN_CONFIG, tokenizer, DEMO_TEXT, seed=SEED)

    print(f"\n  ▶ Entrenamiento D (seed={SEED}):")
    result_d = entrenar_mini(TRAIN_CONFIG, tokenizer, DEMO_TEXT, seed=SEED)

    identicos = comparar_pesos(result_c, result_d, f"C (seed={SEED})", f"D (seed={SEED})")

    if identicos:
        print("\n  💡 EXPLICACIÓN: Al fijar la semilla antes de cada entrenamiento:")
        print("     • Los pesos iniciales son idénticos")
        print("     • El DataLoader baraja los datos en el mismo orden")
        print("     • Dropout desactiva las mismas neuronas")
        print("     → El entrenamiento sigue exactamente el mismo camino.")
        print("     → Los modelos finales son IDÉNTICOS byte a byte.")

    # =====================================================================
    # PARTE 3: Generación de texto — también es aleatoria
    # =====================================================================
    print("\n\n" + "=" * 64)
    print("  PARTE 3: Generación de texto — el sampling también importa")
    print("=" * 64)

    # Usamos un modelo ya entrenado (result_c)
    model = result_c['model']
    prompt = "El dragón "

    print(f"\n  Prompt: \"{prompt}\"")

    print("\n  ▶ Generación sin semilla (3 intentos):")
    for i in range(3):
        texto = demo_generacion(model, tokenizer, prompt, seed=None)
        print(f"    {i+1}. {texto[:70]}...")

    print(f"\n  ▶ Generación con semilla fija (seed=123, 3 intentos):")
    textos_con_seed = []
    for i in range(3):
        texto = demo_generacion(model, tokenizer, prompt, seed=123)
        textos_con_seed.append(texto)
        print(f"    {i+1}. {texto[:70]}...")

    gen_iguales = all(t == textos_con_seed[0] for t in textos_con_seed)
    print(f"\n  Las 3 generaciones con seed son idénticas: {'✅ SÍ' if gen_iguales else '❌ NO'}")

    if gen_iguales:
        print("\n  💡 EXPLICACIÓN: torch.multinomial() usa el generador aleatorio")
        print("     de PyTorch. Al fijar la semilla antes de cada generación,")
        print("     el sampling elige los mismos tokens cada vez.")

    # =====================================================================
    # PARTE 4: Diferentes semillas → diferentes resultados
    # =====================================================================
    print("\n\n" + "=" * 64)
    print("  PARTE 4: Diferentes semillas → diferentes modelos")
    print("=" * 64)

    print(f"\n  ▶ Entrenamiento E (seed=42):")
    result_e = entrenar_mini(TRAIN_CONFIG, tokenizer, DEMO_TEXT, seed=42)

    print(f"\n  ▶ Entrenamiento F (seed=123):")
    result_f = entrenar_mini(TRAIN_CONFIG, tokenizer, DEMO_TEXT, seed=123)

    comparar_pesos(result_e, result_f, "E (seed=42)", "F (seed=123)")

    print("\n  💡 EXPLICACIÓN: Diferentes semillas producen diferentes secuencias")
    print("     de números aleatorios, por lo que los modelos divergen.")
    print("     La semilla NO garantiza que el resultado sea 'bueno',")
    print("     solo que sea REPETIBLE.")

    # =====================================================================
    # RESUMEN
    # =====================================================================
    print("\n\n" + "═" * 64)
    print("  📋 RESUMEN DE REPRODUCIBILIDAD")
    print("═" * 64)
    print("""
  Para reproducir exactamente un entrenamiento necesitas fijar:

    ┌────────────────────────────────┬──────────────────────────────┐
    │ Fuente de aleatoriedad         │ Cómo fijarla                 │
    ├────────────────────────────────┼──────────────────────────────┤
    │ Inicialización de pesos        │ torch.manual_seed(42)        │
    │ DataLoader shuffle             │ DataLoader(generator=g)      │
    │ Dropout                        │ torch.manual_seed(42)        │
    │ Generación (multinomial)       │ torch.manual_seed(42)        │
    │ NumPy (si se usa)              │ np.random.seed(42)           │
    │ Python random                  │ random.seed(42)              │
    │ cuDNN (GPU)                    │ cudnn.deterministic = True   │
    │ Hashing de Python              │ PYTHONHASHSEED=42            │
    └────────────────────────────────┴──────────────────────────────┘

  ⚠️  IMPORTANTE: La reproducibilidad exacta entre CPU y GPU, o entre
  diferentes versiones de PyTorch/CUDA, NO está garantizada. Los
  resultados son reproducibles dentro del MISMO entorno.
""")


if __name__ == "__main__":
    main()
