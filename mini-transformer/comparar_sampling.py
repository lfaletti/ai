"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     COMPARACIÓN VISUAL: CÓMO LA SEMILLA CAMBIA EL TEXTO GENERADO          ║
║                                                                            ║
║  Mismo modelo + mismo prompt + misma temperatura                           ║
║  PERO diferentes semillas → diferentes textos                              ║
║                                                                            ║
║  La semilla controla torch.multinomial() en la línea 902 de                ║
║  mini_transformer.py. Cada semilla produce una secuencia diferente         ║
║  de números pseudoaleatorios, lo que lleva a elegir tokens distintos.      ║
╚══════════════════════════════════════════════════════════════════════════════╝

¿QUÉ HACE torch.manual_seed(N) EXACTAMENTE?
=============================================

PyTorch tiene un generador de números pseudoaleatorios (PRNG) basado en
el algoritmo Mersenne Twister. Funciona así:

  1. torch.manual_seed(N) inicializa el ESTADO INTERNO del generador
     a un valor determinado por N.

  2. Cada vez que se llama a una función aleatoria (multinomial, randn,
     rand, etc.), el generador produce un número y AVANZA su estado.

  3. La SECUENCIA de números es completamente determinista dado un
     estado inicial (semilla).

  Ejemplo:
    seed=42:   0.8823, 0.9150, 0.3829, 0.9593, ...
    seed=0:    0.4963, 0.7682, 0.0885, 0.1320, ...
    seed=123:  0.2961, 0.5166, 0.2517, 0.6886, ...

  Cada secuencia es diferente, pero SIEMPRE la misma para la misma semilla.

IMPACTO EN LA GENERACIÓN:
=========================

  Con seed=42, el primer multinomial podría "tirar" 0.88 → elige token A
  Con seed=0,  el primer multinomial podría "tirar" 0.49 → elige token B

  En el segundo paso, los contextos ya son diferentes (uno tiene A, otro B),
  así que las probabilidades son diferentes, y además el generador da otro
  número diferente. Los textos DIVERGEN rápidamente.
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_transformer import MiniTransformer, CharTokenizer


# =============================================================================
# Cargar modelo
# =============================================================================
def cargar_modelo(model_path='modelo_entrenado.pt'):
    """Carga el modelo y tokenizer. Crea uno de demo si no existe."""
    if os.path.exists(model_path):
        model = MiniTransformer.load(model_path, device='cpu')
        tokenizer = CharTokenizer()
        tokenizer.token_to_id = model.token_to_id
        tokenizer.id_to_token = model.id_to_token
        tokenizer.vocab_size = model.vocab_size
        return model, tokenizer
    else:
        print(f"⚠️  {model_path} no encontrado. Creando modelo de demo...")
        return _crear_modelo_rapido()


def _crear_modelo_rapido():
    """Crea un modelo mínimo para demo."""
    import torch.nn as nn
    from mini_transformer import TextDataset
    from torch.utils.data import DataLoader

    text = ("El dragón leía libros bajo las estrellas. "
            "La exploradora buscaba aventuras. "
            "Los niños escuchaban historias del dragón sabio. ") * 8
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    torch.manual_seed(42)
    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size, d_model=32, num_heads=2,
        num_layers=1, d_ff=64, max_seq_len=64, dropout=0.1,
        token_to_id=tokenizer.token_to_id, id_to_token=tokenizer.id_to_token,
    )
    dataset = TextDataset(text, tokenizer, seq_len=32)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(10):
        for x, y in loader:
            loss = criterion(model(x)[0].view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model, tokenizer


# =============================================================================
# Generar con semilla específica y capturar decisiones token a token
# =============================================================================
def generar_con_detalle(model, tokenizer, prompt, max_tokens=30,
                         temperature=0.7, seed=None):
    """
    Genera texto capturando las decisiones de cada paso.

    Returns:
        texto: str — texto generado completo
        decisiones: list[dict] — info de cada paso del multinomial
    """
    if seed is not None:
        # ┌──────────────────────────────────────────────────────────┐
        # │ >>> torch.manual_seed(seed) <<<                          │
        # │                                                          │
        # │ Esto resetea el generador pseudoaleatorio de PyTorch.    │
        # │ A partir de aquí, TODA operación aleatoria (incluyendo   │
        # │ torch.multinomial en la línea 902 de mini_transformer.py)│
        # │ producirá la MISMA secuencia de números.                 │
        # └──────────────────────────────────────────────────────────┘
        torch.manual_seed(seed)

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        return prompt, []

    model.eval()
    generated = torch.tensor([prompt_ids], dtype=torch.long)
    decisiones = []

    with torch.no_grad():
        for step in range(max_tokens):
            context = generated[:, -model.max_seq_len:]

            # Forward pass (DETERMINISTA dado los mismos inputs)
            logits, _ = model(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Softmax (DETERMINISTA)
            probs = F.softmax(next_logits, dim=-1)

            # Top-3 candidatos antes de tirar el dado
            top_probs, top_ids = torch.topk(probs[0], min(3, probs.size(-1)))
            candidatos = []
            for p, idx in zip(top_probs, top_ids):
                ch = tokenizer.id_to_token.get(idx.item(), '?')
                candidatos.append({'char': ch, 'prob': p.item(), 'id': idx.item()})

            # ┌──────────────────────────────────────────────────────┐
            # │  🎲🎲🎲  AQUÍ SE TIRA EL DADO  🎲🎲🎲              │
            # │                                                      │
            # │  torch.multinomial consume UN número del generador   │
            # │  aleatorio y elige un índice según 'probs'.          │
            # │                                                      │
            # │  Con seed=42: siempre elige la misma secuencia       │
            # │  Con seed=0:  elige una secuencia diferente          │
            # │  Sin seed:    cada ejecución es impredecible         │
            # └──────────────────────────────────────────────────────┘
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            token_char = tokenizer.id_to_token.get(token_id, '?')
            token_prob = probs[0, token_id].item()

            decisiones.append({
                'step': step,
                'elegido': token_char,
                'elegido_prob': token_prob,
                'candidatos': candidatos,
            })

            generated = torch.cat([generated, next_token], dim=1)

    texto = prompt + ''.join(d['elegido'] for d in decisiones)
    return texto, decisiones


# =============================================================================
# Visualización lado a lado
# =============================================================================
def mostrar_lado_a_lado(resultados, prompt, max_display=60):
    """
    Muestra los textos generados lado a lado con formato visual.
    """
    print(f"\n  {'─' * 72}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  {'─' * 72}")

    for seed, texto, decisiones in resultados:
        # Formatear el texto generado (solo la parte nueva)
        parte_nueva = texto[len(prompt):]
        display = parte_nueva[:max_display]
        display_show = display.replace('\n', '↵').replace(' ', '·') if len(display) < 20 else display

        print(f"\n  seed={seed:>3d}  │ \"{prompt}{display}{'...' if len(parte_nueva) > max_display else ''}\"")

        # Mostrar primeras 8 decisiones del dado
        if decisiones:
            tokens_show = []
            for d in decisiones[:8]:
                ch = d['elegido']
                if ch == ' ':
                    ch = '␣'
                elif ch == '\n':
                    ch = '↵'
                p = d['elegido_prob']
                tokens_show.append(f"{ch}({p*100:.0f}%)")
            print(f"           │ Decisiones del dado: {' → '.join(tokens_show)}{'...' if len(decisiones) > 8 else ''}")


def mostrar_divergencia(resultados, prompt):
    """
    Muestra en qué paso los textos empiezan a divergir entre sí.
    """
    if len(resultados) < 2:
        return

    print(f"\n  {'─' * 72}")
    print(f"  📊 Análisis de divergencia")
    print(f"  {'─' * 72}")

    base_seed, base_texto, base_dec = resultados[0]
    for seed, texto, dec in resultados[1:]:
        # Encontrar primer punto de divergencia
        diverge_at = None
        for i, (d1, d2) in enumerate(zip(base_dec, dec)):
            if d1['elegido'] != d2['elegido']:
                diverge_at = i
                break

        if diverge_at is not None:
            d1 = base_dec[diverge_at]
            d2 = dec[diverge_at]
            ch1 = d1['elegido'] if d1['elegido'] not in (' ', '\n') else ('␣' if d1['elegido'] == ' ' else '↵')
            ch2 = d2['elegido'] if d2['elegido'] not in (' ', '\n') else ('␣' if d2['elegido'] == ' ' else '↵')
            print(f"\n  seed={base_seed} vs seed={seed}: divergen en el paso {diverge_at}")
            print(f"    seed={base_seed}: eligió '{ch1}' (prob: {d1['elegido_prob']*100:.1f}%)")
            print(f"    seed={seed:>3d}: eligió '{ch2}' (prob: {d2['elegido_prob']*100:.1f}%)")
            # Mostrar candidatos de ese paso
            print(f"    Candidatos disponibles (seed={base_seed}):")
            for c in d1['candidatos']:
                ch = c['char'] if c['char'] not in (' ', '\n') else ('␣' if c['char'] == ' ' else '↵')
                print(f"      '{ch}': {c['prob']*100:.1f}%")
            print(f"    💡 Ambos ven las MISMAS probabilidades, pero multinomial()")
            print(f"       recibe un número aleatorio diferente → elige diferente.")
        else:
            min_len = min(len(base_dec), len(dec))
            if min_len > 0:
                print(f"\n  seed={base_seed} vs seed={seed}: ¡coinciden en los {min_len} pasos comparados!")
            else:
                print(f"\n  seed={base_seed} vs seed={seed}: sin decisiones para comparar.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("╔" + "═" * 72 + "╗")
    print("║  COMPARACIÓN VISUAL: CÓMO LA SEMILLA CAMBIA EL TEXTO GENERADO       ║")
    print("╚" + "═" * 72 + "╝")

    model, tokenizer = cargar_modelo()

    # =====================================================================
    # EXPERIMENTO 1: Mismo prompt, diferentes semillas
    # =====================================================================
    prompt = "El dragón "
    semillas = [0, 1, 2, 42, 100]

    print(f"\n{'=' * 74}")
    print(f"  🧪 EXPERIMENTO 1: Mismo prompt, {len(semillas)} semillas diferentes")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Semillas: {semillas}")
    print(f"  Temperatura: 0.7")
    print(f"{'=' * 74}")

    resultados = []
    for seed in semillas:
        texto, decisiones = generar_con_detalle(
            model, tokenizer, prompt,
            max_tokens=40, temperature=0.7, seed=seed
        )
        resultados.append((seed, texto, decisiones))

    mostrar_lado_a_lado(resultados, prompt)
    mostrar_divergencia(resultados, prompt)

    # =====================================================================
    # EXPERIMENTO 2: Efecto de la temperatura con misma semilla
    # =====================================================================
    print(f"\n\n{'=' * 74}")
    print(f"  🌡️ EXPERIMENTO 2: Misma semilla (42), diferentes temperaturas")
    print(f"  Temperatura afecta las probabilidades ANTES del dado")
    print(f"{'=' * 74}")
    print(f"""
  Recordar: la temperatura se aplica ANTES de multinomial():

    T=0.3 → distribución "puntiaguda" → pocos candidatos viables
             multinomial casi siempre elige el más probable
    T=0.7 → distribución balanceada → competencia moderada
    T=1.5 → distribución "plana" → muchos candidatos viables
             multinomial puede elegir tokens poco probables
""")

    temperaturas = [0.3, 0.7, 1.0, 1.5]
    for temp in temperaturas:
        torch.manual_seed(42)  # Misma semilla para todos
        texto, dec = generar_con_detalle(
            model, tokenizer, prompt,
            max_tokens=40, temperature=temp, seed=42
        )
        parte_nueva = texto[len(prompt):][:55]
        print(f"  T={temp:.1f}  │ \"{prompt}{parte_nueva}...\"")

    print(f"\n  💡 Misma semilla (42) pero diferentes temperaturas → textos diferentes.")
    print(f"     La temperatura cambia las PROBABILIDADES, así que multinomial()")
    print(f"     ve una distribución diferente aunque el número aleatorio sea el mismo.")

    # =====================================================================
    # EXPERIMENTO 3: Verificación de reproducibilidad
    # =====================================================================
    print(f"\n\n{'=' * 74}")
    print(f"  🔁 EXPERIMENTO 3: Verificación — cada semilla es reproducible")
    print(f"{'=' * 74}")

    print(f"\n  Generando 2 veces con cada semilla para confirmar identidad:\n")
    for seed in [0, 42, 100]:
        texto1, _ = generar_con_detalle(model, tokenizer, prompt, 30, 0.7, seed=seed)
        texto2, _ = generar_con_detalle(model, tokenizer, prompt, 30, 0.7, seed=seed)
        iguales = texto1 == texto2
        parte = texto1[len(prompt):][:40]
        print(f"  seed={seed:>3d}: {'✅ idénticos' if iguales else '❌ diferentes'}"
              f"  │ \"{prompt}{parte}...\"")

    # =====================================================================
    # DIAGRAMA: Cómo la semilla fluye al multinomial
    # =====================================================================
    print(f"\n\n{'═' * 74}")
    print(f"  📐 DIAGRAMA: Flujo de la semilla al resultado final")
    print(f"{'═' * 74}")
    print(f"""
  torch.manual_seed(42)
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Generador Pseudoaleatorio (Mersenne Twister)           │
  │  Estado interno: determinado por seed=42                │
  │                                                         │
  │  Secuencia que produce:                                 │
  │    0.8823 → 0.9150 → 0.3829 → 0.9593 → ...            │
  └──────────────────────┬──────────────────────────────────┘
                         │
    Para cada token nuevo:│
                         ▼
  ┌──────────────────────────────────────────┐
  │  Modelo calcula probabilidades:          │
  │    "a"=0.35, "e"=0.25, "s"=0.20, ...    │  ← DETERMINISTA
  └──────────────────────┬───────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────┐
  │  torch.multinomial(probs, 1)             │
  │                                          │
  │  Consume el SIGUIENTE número del         │  ← 🎲 AQUÍ
  │  generador (ej: 0.8823)                  │
  │                                          │
  │  Con 0.8823 y probs=[0.35, 0.25, 0.20]  │
  │  → Acumulada: [0.35, 0.60, 0.80, 1.0]   │
  │  → 0.8823 cae en el rango [0.80, 1.0]   │
  │  → Elige el 4to token                    │
  └──────────────────────┬───────────────────┘
                         │
                         ▼
                Token elegido + se repite
                (el generador avanza al 0.9150 para el siguiente token)
""")

    print(f"  🎯 Resumen: torch.multinomial() en la línea 902 de mini_transformer.py")
    print(f"     es la ÚNICA fuente de aleatoriedad durante la generación.")
    print(f"     torch.manual_seed(N) antes de generate() controla completamente")
    print(f"     qué texto se produce.\n")


if __name__ == "__main__":
    main()
