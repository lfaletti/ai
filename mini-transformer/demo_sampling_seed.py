"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         DEMO: DÓNDE ESTÁ EL "DADO" EN LA GENERACIÓN DE TEXTO              ║
║                                                                            ║
║  Este script localiza y demuestra la ÚNICA LÍNEA de código que hace        ║
║  que la generación de texto sea aleatoria: torch.multinomial()             ║
║                                                                            ║
║  Archivo: mini_transformer.py                                              ║
║  Función: MiniTransformer.generate()                                       ║
║  Línea:   902                                                              ║
║                                                                            ║
║  >>> next_token = torch.multinomial(probs, num_samples=1)  <<<             ║
║                                                                            ║
║  Esta es la tirada de dado. Todo lo demás es DETERMINISTA.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

¿CÓMO FUNCIONA torch.multinomial?
==================================

torch.multinomial(probs, num_samples=1) hace esto:

  1. Recibe un vector de probabilidades, ej: [0.4, 0.3, 0.2, 0.1]
     (que representan tokens como: ["el", "un", "la", "su"])

  2. "Tira un dado" cargado con esas probabilidades:
     - 40% de chance de elegir "el"   (índice 0)
     - 30% de chance de elegir "un"   (índice 1)
     - 20% de chance de elegir "la"   (índice 2)
     - 10% de chance de elegir "su"   (índice 3)

  3. Devuelve el ÍNDICE del token elegido (ej: tensor([0]) → "el")

El resultado depende del GENERADOR ALEATORIO INTERNO de PyTorch.
Ese generador es una secuencia de números pseudoaleatorios determinada
por una SEMILLA (seed). Misma semilla → misma secuencia → misma elección.

PIPELINE COMPLETO DE GENERACIÓN (para un token):
=================================================

  Contexto actual: "El dragón leía"
       │
       ▼ [DETERMINISTA] Forward pass del modelo
  Logits: [2.1, -0.5, 1.8, 0.3, ...]  (uno por cada token del vocabulario)
       │
       ▼ [DETERMINISTA] Dividir por temperatura
  Logits escalados: [2.63, -0.63, 2.25, 0.38, ...]
       │
       ▼ [DETERMINISTA] Filtrar con top-k / top-p
  Logits filtrados: [2.63, -inf, 2.25, -inf, ...]
       │
       ▼ [DETERMINISTA] Softmax → probabilidades
  Probs: [0.41, 0.0, 0.35, 0.0, ..., 0.24]
       │
       ▼ [🎲 ALEATORIO] torch.multinomial(probs, 1)  ← ¡AQUÍ ESTÁ EL DADO!
  Token elegido: índice 2  →  "libros"
       │
       ▼
  Nuevo contexto: "El dragón leía libros"
  (se repite el proceso para el siguiente token)
"""

import os
import sys
import torch

# Asegurar que podemos importar desde el directorio del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mini_transformer import MiniTransformer, CharTokenizer


# =============================================================================
# FUNCIÓN AUXILIAR: Cargar el modelo entrenado
# =============================================================================
def cargar_modelo(model_path: str = 'modelo_entrenado.pt'):
    """
    Carga el modelo entrenado y reconstruye el tokenizer.

    Si no existe el modelo entrenado, crea uno pequeño de demostración
    entrenándolo rápidamente con unos pocos pasos.
    """
    device = 'cpu'

    if os.path.exists(model_path):
        print(f"📂 Cargando modelo desde: {model_path}")
        model = MiniTransformer.load(model_path, device=device)
        # Reconstruir tokenizer desde el modelo
        tokenizer = CharTokenizer()
        tokenizer.token_to_id = model.token_to_id
        tokenizer.id_to_token = model.id_to_token
        tokenizer.vocab_size = model.vocab_size
        return model, tokenizer
    else:
        print(f"⚠️  No se encontró {model_path}")
        print(f"   Creando un modelo de demostración rápido...")
        return _crear_modelo_demo()


def _crear_modelo_demo():
    """Crea y entrena un modelo mínimo si no hay uno guardado."""
    import torch.nn as nn
    from mini_transformer import TextDataset
    from torch.utils.data import DataLoader

    text = ("El dragón leía libros bajo las estrellas brillantes. "
            "Cada noche contaba constelaciones desde la montaña. "
            "Una exploradora llegó buscando aventuras y conocimiento. "
            "Los niños escuchaban historias maravillosas del dragón sabio. ") * 5

    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    torch.manual_seed(42)
    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32, num_heads=2, num_layers=1, d_ff=64,
        max_seq_len=64, dropout=0.1,
        token_to_id=tokenizer.token_to_id,
        id_to_token=tokenizer.id_to_token,
    )

    dataset = TextDataset(text, tokenizer, seq_len=32)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for x, y in loader:
            logits, _ = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"   ✅ Modelo de demostración creado ({model.count_parameters():,} params)")
    return model, tokenizer


# =============================================================================
# FUNCIÓN: Generar texto mostrando el "dado" en acción
# =============================================================================
def generar_mostrando_dado(model, tokenizer, prompt: str,
                            max_tokens: int = 30, temperature: float = 0.7,
                            seed: int = None, verbose: bool = True):
    """
    Genera texto token por token, mostrando exactamente qué pasa
    en cada tirada del "dado" (torch.multinomial).

    Esta función REPLICA la lógica de MiniTransformer.generate()
    pero añade prints educativos en cada paso.
    """
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)
        if verbose:
            print(f"  🌱 Semilla fijada: {seed}")
    else:
        if verbose:
            print(f"  🎲 Sin semilla (aleatorio puro)")

    # Codificar el prompt
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        print("  ⚠️  El prompt no contiene caracteres reconocidos.")
        return ""

    model.eval()
    generated = torch.tensor([prompt_ids], dtype=torch.long)

    tokens_generados = []

    with torch.no_grad():
        for step in range(max_tokens):
            context = generated[:, -model.max_seq_len:]

            # ══════════════════════════════════════════════════════════
            # PASO 1: Forward pass (DETERMINISTA)
            # El modelo calcula logits para cada posible siguiente token
            # ══════════════════════════════════════════════════════════
            logits, _ = model(context)
            next_logits = logits[:, -1, :]  # Solo el último token

            # ══════════════════════════════════════════════════════════
            # PASO 2: Aplicar temperatura (DETERMINISTA)
            # Dividir por T reshapea la distribución
            # ══════════════════════════════════════════════════════════
            next_logits = next_logits / max(temperature, 1e-8)

            # ══════════════════════════════════════════════════════════
            # PASO 3: Softmax → probabilidades (DETERMINISTA)
            # Convierte logits en probabilidades que suman 1
            # ══════════════════════════════════════════════════════════
            probs = F.softmax(next_logits, dim=-1)

            # ══════════════════════════════════════════════════════════
            # PASO 4: 🎲🎲🎲 torch.multinomial — ¡EL DADO! 🎲🎲🎲
            #
            # >>> ESTA ES LA LÍNEA QUE HACE TODO ALEATORIO <<<
            #
            # Archivo: mini_transformer.py, línea 902
            # Función: MiniTransformer.generate()
            #
            # Lo que hace:
            #   - Recibe el vector 'probs' (probabilidades de cada token)
            #   - Usa el generador aleatorio interno de PyTorch
            #   - Elige UN índice según esas probabilidades
            #   - El generador avanza su estado interno
            #
            # Si la semilla es la misma, el generador produce la misma
            # secuencia de números → elige los mismos tokens → mismo texto.
            # ══════════════════════════════════════════════════════════
            next_token = torch.multinomial(probs, num_samples=1)
            # ════════════════════ FIN DEL DADO ═══════════════════════

            token_id = next_token.item()
            token_char = tokenizer.id_to_token.get(token_id, '?')
            token_prob = probs[0, token_id].item()

            tokens_generados.append(token_char)

            # Mostrar detalle de los primeros pasos
            if verbose and step < 5:
                # Top-3 candidatos
                top_probs, top_ids = torch.topk(probs[0], min(3, probs.size(-1)))
                candidatos = [
                    f"'{tokenizer.id_to_token.get(idx.item(), '?')}' ({p.item()*100:.1f}%)"
                    for p, idx in zip(top_probs, top_ids)
                ]
                display_char = token_char if token_char not in (' ', '\n') else \
                    ('␣' if token_char == ' ' else '↵')
                print(f"    Paso {step}: Top candidatos: {', '.join(candidatos)} "
                      f"→ Elegido: '{display_char}' ({token_prob*100:.1f}%)")

            generated = torch.cat([generated, next_token], dim=1)

    texto_completo = prompt + ''.join(tokens_generados)

    if verbose and max_tokens > 5:
        print(f"    ... ({max_tokens - 5} pasos más)")

    return texto_completo


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
def main():
    print("╔" + "═" * 66 + "╗")
    print("║  DEMO: DÓNDE ESTÁ EL \"DADO\" EN LA GENERACIÓN DE TEXTO           ║")
    print("╚" + "═" * 66 + "╝")

    model, tokenizer = cargar_modelo()
    prompt = "El dragón "

    # =====================================================================
    # PARTE 1: Ubicar el "dado" en el código
    # =====================================================================
    print("\n" + "=" * 68)
    print("  📍 PARTE 1: Ubicación exacta del \"dado\" en el código")
    print("=" * 68)
    print("""
  Archivo:   mini_transformer.py
  Clase:     MiniTransformer
  Método:    generate()
  Línea:     902

  El código relevante es:

  ┌─────────────────────────────────────────────────────────────────┐
  │  # línea 900: Convertir logits a probabilidades y samplear     │
  │  probs = F.softmax(next_token_logits, dim=-1)                  │
  │                                                                 │
  │  # línea 902: ⬇⬇⬇ AQUÍ SE TIRA EL DADO ⬇⬇⬇                  │
  │  next_token = torch.multinomial(probs, num_samples=1)          │
  │  #             ^^^^^^^^^^^^^^^^                                 │
  │  #             Esta función USA el generador aleatorio          │
  │  #             interno de PyTorch para elegir UN token          │
  │  #             según las probabilidades calculadas.             │
  └─────────────────────────────────────────────────────────────────┘

  También hay un segundo multinomial en generate_with_trace() (línea 943),
  que se usa para la visualización — funciona exactamente igual.

  TODO LO DEMÁS en generate() es DETERMINISTA:
    ✅ Forward pass (logits = model(context))     → siempre igual
    ✅ Temperatura (logits / T)                    → siempre igual
    ✅ Top-k filtering                             → siempre igual
    ✅ Top-p filtering                             → siempre igual
    ✅ Softmax (probs = softmax(logits))           → siempre igual
    🎲 multinomial(probs) ← ÚNICO PUNTO ALEATORIO → depende de la semilla
""")

    # =====================================================================
    # PARTE 2: Generar SIN semilla → resultados diferentes cada vez
    # =====================================================================
    print("=" * 68)
    print("  🎲 PARTE 2: Generación SIN semilla (3 intentos)")
    print("  Cada llamada a multinomial usa números aleatorios diferentes")
    print("=" * 68)

    textos_sin_seed = []
    for i in range(3):
        print(f"\n  ── Intento {i+1} ──")
        texto = generar_mostrando_dado(
            model, tokenizer, prompt,
            max_tokens=40, temperature=0.7,
            seed=None, verbose=(i == 0)  # Solo verbose en el primero
        )
        textos_sin_seed.append(texto)
        print(f"  Resultado: \"{texto[:80]}...\"")

    # Verificar que son diferentes
    todos_diferentes = len(set(textos_sin_seed)) == len(textos_sin_seed)
    print(f"\n  ¿Los 3 textos son diferentes entre sí? "
          f"{'✅ SÍ' if todos_diferentes else '⚠️ Coincidencia (posible pero improbable)'}")
    print(f"  💡 Porque cada vez que multinomial() se ejecuta, el generador")
    print(f"     aleatorio está en un estado diferente → elige tokens distintos.")

    # =====================================================================
    # PARTE 3: Generar CON semilla fija → resultados idénticos
    # =====================================================================
    SEED = 42

    print(f"\n\n{'=' * 68}")
    print(f"  🌱 PARTE 3: Generación CON semilla fija (seed={SEED}, 3 intentos)")
    print(f"  torch.manual_seed({SEED}) resetea el generador al MISMO estado")
    print(f"{'=' * 68}")

    textos_con_seed = []
    for i in range(3):
        print(f"\n  ── Intento {i+1} ──")
        texto = generar_mostrando_dado(
            model, tokenizer, prompt,
            max_tokens=40, temperature=0.7,
            seed=SEED, verbose=(i == 0)  # Solo verbose en el primero
        )
        textos_con_seed.append(texto)
        print(f"  Resultado: \"{texto[:80]}...\"")

    # Verificar que son idénticos
    todos_iguales = len(set(textos_con_seed)) == 1
    print(f"\n  ¿Los 3 textos son idénticos? {'✅ SÍ' if todos_iguales else '❌ NO'}")
    if todos_iguales:
        print(f"  💡 Porque torch.manual_seed({SEED}) pone el generador en el")
        print(f"     MISMO estado inicial cada vez. multinomial() recibe los")
        print(f"     mismos números pseudoaleatorios → elige los mismos tokens.")

    # =====================================================================
    # PARTE 4: Demostración directa de torch.multinomial
    # =====================================================================
    print(f"\n\n{'=' * 68}")
    print(f"  🔬 PARTE 4: torch.multinomial al desnudo")
    print(f"{'=' * 68}")
    print(f"""
  Para entender el "dado", veamos multinomial aislado:

  Supongamos que el modelo calculó estas probabilidades para el
  siguiente token:
    "a" → 40%,  "e" → 30%,  "o" → 20%,  "i" → 10%
""")

    probs_ejemplo = torch.tensor([0.4, 0.3, 0.2, 0.1])
    opciones = ["a", "e", "o", "i"]

    print(f"  Sin semilla (10 tiradas):")
    resultados = []
    for _ in range(10):
        idx = torch.multinomial(probs_ejemplo, 1).item()
        resultados.append(opciones[idx])
    print(f"    {' '.join(resultados)}")
    print(f"    → Cada tirada puede dar diferente (pero 'a' sale más seguido)")

    print(f"\n  Con seed=42 (3 rondas de 10 tiradas):")
    for ronda in range(3):
        torch.manual_seed(42)  # ← Resetear el dado al mismo estado
        resultados = []
        for _ in range(10):
            idx = torch.multinomial(probs_ejemplo, 1).item()
            resultados.append(opciones[idx])
        print(f"    Ronda {ronda+1}: {' '.join(resultados)}")
    print(f"    → ¡Las 3 rondas son IDÉNTICAS! Misma semilla = mismo resultado.")

    # =====================================================================
    # RESUMEN
    # =====================================================================
    print(f"\n\n{'═' * 68}")
    print(f"  📋 RESUMEN")
    print(f"{'═' * 68}")
    print(f"""
  🎯 El "dado" de la generación de texto está en UNA sola línea:

     mini_transformer.py, línea 902:
     next_token = torch.multinomial(probs, num_samples=1)

  🔧 Para controlarlo:
     torch.manual_seed(N)   →  antes de llamar a model.generate()

  📌 Notas importantes:
     • La semilla se consume con cada llamada a multinomial.
       Si generas 50 tokens, el generador avanza 50 pasos.
     • Fijar la semilla ANTES de generate() garantiza que los 50
       tokens sean idénticos entre ejecuciones.
     • Cambiar CUALQUIER cosa (prompt, temperatura, top_k, top_p)
       cambiará las probabilidades → cambia el texto resultante
       incluso con la misma semilla.
""")


if __name__ == "__main__":
    main()
