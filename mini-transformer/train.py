"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SCRIPT DE ENTRENAMIENTO DEL MINI TRANSFORMER EDUCATIVO           ║
║                                                                            ║
║  Este script:                                                              ║
║    1. Prepara un dataset de texto pequeño                                  ║
║    2. Entrena el modelo Mini Transformer                                   ║
║    3. Genera texto nuevo con el modelo entrenado                           ║
║    4. Exporta datos de trazas para la visualización interactiva            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mini_transformer import MiniTransformer, CharTokenizer, TextDataset


# =============================================================================
# DATASET DE ENTRENAMIENTO
# =============================================================================
# Un mini-corpus de cuentos y frases en español para entrenar.
# Es pequeño a propósito — suficiente para que el modelo aprenda
# patrones básicos del lenguaje y demuestre que funciona.

TRAINING_TEXT = """
Érase una vez un pequeño dragón que vivía en lo alto de una montaña.
El dragón no era como los demás dragones. No le gustaba asustar a la gente
ni quemar aldeas. En cambio, le encantaba leer libros y contar estrellas.

Cada noche, el dragón subía a la cima de la montaña y miraba el cielo.
Las estrellas brillaban como diamantes sobre un manto de terciopelo negro.
El dragón las contaba una por una, inventando historias para cada constelación.

Un día, una joven exploradora llegó a la montaña buscando aventuras.
Cuando vio al dragón, no sintió miedo. El dragón tenía un libro abierto
entre sus garras y unas pequeñas gafas redondas sobre su hocico.

La exploradora se acercó con curiosidad y le preguntó qué leía.
El dragón respondió que era un libro sobre las estrellas y los planetas.
Desde ese día, la exploradora y el dragón se hicieron grandes amigos.

Cada noche se sentaban juntos en la cima de la montaña a observar las estrellas.
El dragón le enseñaba los nombres de las constelaciones y la exploradora
le contaba historias sobre los lugares lejanos que había visitado.

Los habitantes de la aldea cercana, al principio, tenían miedo del dragón.
Pero cuando vieron que era amable y sabio, empezaron a visitarlo también.
El dragón se convirtió en el maestro más querido de toda la región.

Los niños subían a la montaña para escuchar sus historias sobre estrellas
y mundos lejanos. El dragón les enseñaba sobre el universo, la naturaleza
y la importancia de ser curiosos y amables con todos los seres.

Con el paso del tiempo, la montaña se llenó de vida y alegría.
Los pájaros cantaban al amanecer, las flores crecían por todas partes,
y el pequeño dragón sabio vivía feliz rodeado de amigos y libros.

La historia del dragón lector se extendió por todo el reino.
Gente de todas partes viajaba para conocerlo y aprender de su sabiduría.
Y así, el dragón que era diferente se convirtió en el más especial de todos.

Porque a veces, ser diferente es lo que te hace verdaderamente extraordinario.
Y la curiosidad, combinada con la bondad, puede transformar el mundo entero.

El pequeño dragón aprendió que no necesitaba fuego para brillar.
Su luz venía de dentro, de su amor por el conocimiento y por los demás.
Y esa luz era más poderosa que cualquier llama que pudiera escupir.

Así termina esta historia, pero el dragón sigue allí, en su montaña,
leyendo libros bajo las estrellas, esperando la visita de nuevos amigos
que quieran compartir el maravilloso regalo del conocimiento.
"""


def train_model(
    text: str = TRAINING_TEXT,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    d_ff: int = 512,
    seq_len: int = 64,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 3e-4,
    device: str = 'auto',
    save_path: str = 'modelo_entrenado.pt',
    trace_path: str = 'trace_data.json',
):
    """
    Entrena el Mini Transformer en un texto dado.

    Args:
        text: Texto de entrenamiento
        d_model: Dimensión del modelo
        num_heads: Número de cabezas de atención
        num_layers: Número de capas
        d_ff: Dimensión de la capa intermedia FFN
        seq_len: Longitud de secuencia para entrenamiento
        batch_size: Tamaño del batch
        num_epochs: Número de épocas de entrenamiento
        learning_rate: Tasa de aprendizaje
        device: 'cpu', 'cuda', o 'auto'
        save_path: Ruta para guardar el modelo
        trace_path: Ruta para guardar datos de visualización
    """
    # =====================================================================
    # PASO 1: Configurar dispositivo
    # =====================================================================
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Dispositivo: {device}")
    print(f"{'=' * 60}")

    # =====================================================================
    # PASO 2: Preparar el tokenizer y el dataset
    # =====================================================================
    print("\n📚 Preparando datos de entrenamiento...")

    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    dataset = TextDataset(text, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Descartar último batch si es incompleto
    )

    print(f"   Longitud del texto: {len(text):,} caracteres")
    print(f"   Vocabulario: {tokenizer.vocab_size} tokens")
    print(f"   Muestras de entrenamiento: {len(dataset):,}")
    print(f"   Batches por época: {len(dataloader)}")

    # =====================================================================
    # PASO 3: Crear el modelo
    # =====================================================================
    print(f"\n🏗️  Creando modelo...")

    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=seq_len * 2,  # Permitir generación más larga
        dropout=0.1,
        token_to_id=tokenizer.token_to_id,
        id_to_token=tokenizer.id_to_token,
    ).to(device)

    print(model)
    print(f"   Parámetros totales: {model.count_parameters():,}")

    # =====================================================================
    # PASO 4: Configurar optimización
    # =====================================================================
    # AdamW: Adam con weight decay correcto (regularización L2)
    # Es el optimizador estándar para Transformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),  # Momentos de Adam (estándar para Transformers)
        eps=1e-9,
        weight_decay=0.01,  # Regularización para evitar sobreajuste
    )

    # Learning rate scheduler con warmup
    # El warmup incrementa gradualmente el LR al inicio, evitando
    # actualizaciones destructivas cuando los pesos son aleatorios
    warmup_steps = min(100, len(dataloader) * 2)
    total_steps = len(dataloader) * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # Warmup lineal
        # Cosine decay después del warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # CrossEntropyLoss: la función de pérdida estándar para clasificación
    # Compara las probabilidades predichas con el token correcto
    criterion = nn.CrossEntropyLoss()

    # =====================================================================
    # PASO 5: Bucle de entrenamiento
    # =====================================================================
    print(f"\n🚀 Iniciando entrenamiento ({num_epochs} épocas)...")
    print(f"{'=' * 60}")

    training_history = []
    best_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # inputs:  (batch_size, seq_len) - secuencia de entrada
            # targets: (batch_size, seq_len) - secuencia objetivo (desplazada +1)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            # logits: (batch_size, seq_len, vocab_size)
            logits, _ = model(inputs)

            # Reshape para CrossEntropyLoss:
            # logits:  (batch_size × seq_len, vocab_size)
            # targets: (batch_size × seq_len,)
            loss = criterion(
                logits.view(-1, tokenizer.vocab_size),
                targets.view(-1)
            )

            # Backward pass + actualización de pesos
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping: evita explosión de gradientes
            # Si los gradientes son muy grandes, los escala a max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        # Estadísticas de la época
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        # Perplexity: e^loss — mide cuán "confundido" está el modelo
        # Perplexity = 10 significa que el modelo está tan confundido como
        # si estuviera eligiendo entre 10 opciones equiprobables
        perplexity = math.exp(min(avg_loss, 20))  # Cap para evitar overflow

        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'perplexity': perplexity,
            'lr': current_lr,
            'time': elapsed,
        })

        # Imprimir progreso cada 5 épocas o en la primera/última
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            bar_len = int(30 * (epoch + 1) / num_epochs)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(
                f"  Época {epoch+1:3d}/{num_epochs} [{bar}] "
                f"Loss: {avg_loss:.4f} | PPL: {perplexity:.1f} | "
                f"LR: {current_lr:.6f} | {elapsed:.1f}s"
            )

            # Generar muestra de texto cada 10 épocas
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                model.eval()
                sample_prompt = "El dragón "
                prompt_ids = torch.tensor(
                    [tokenizer.encode(sample_prompt)], dtype=torch.long
                ).to(device)
                generated_ids = model.generate(
                    prompt_ids, max_new_tokens=80, temperature=0.7
                )
                generated_text = tokenizer.decode(generated_ids[0].tolist())
                print(f"  📝 Muestra: \"{generated_text[:120]}...\"")
                model.train()

        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss

    # =====================================================================
    # PASO 6: Guardar modelo entrenado
    # =====================================================================
    print(f"\n{'=' * 60}")
    print(f"💾 Guardando modelo...")
    model.save(save_path)

    # =====================================================================
    # PASO 7: Generar texto final y traza para visualización
    # =====================================================================
    print(f"\n{'=' * 60}")
    print(f"✨ Generando texto con el modelo entrenado:\n")

    model.eval()
    prompts = [
        "Érase una vez ",
        "El dragón ",
        "Las estrellas ",
        "La exploradora ",
        "Los niños ",
    ]

    generated_texts = []
    for prompt in prompts:
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long
        ).to(device)
        gen_ids = model.generate(
            prompt_ids, max_new_tokens=150, temperature=0.7, top_p=0.9
        )
        text_gen = tokenizer.decode(gen_ids[0].tolist())
        generated_texts.append(text_gen)
        print(f"  Prompt: \"{prompt}\"")
        print(f"  → {text_gen[:200]}")
        print()

    # =====================================================================
    # PASO 8: Exportar datos para visualización
    # =====================================================================
    print(f"📊 Generando datos de traza para visualización...")

    # Generar traza detallada
    trace_prompt = "El dragón "
    prompt_ids = torch.tensor(
        [tokenizer.encode(trace_prompt)], dtype=torch.long
    ).to(device)
    trace_data = model.generate_with_trace(
        prompt_ids, max_new_tokens=15, temperature=0.7
    )

    # Añadir metadatos
    trace_data['tokenizer'] = tokenizer.to_dict()
    trace_data['training_history'] = training_history
    trace_data['generated_texts'] = [
        {'prompt': p, 'generated': t}
        for p, t in zip(prompts, generated_texts)
    ]
    trace_data['model_info'] = {
        'total_parameters': model.count_parameters(),
        'vocab_size': tokenizer.vocab_size,
        'd_model': model._config['d_model'],
        'num_heads': model._config['num_heads'],
        'num_layers': model._config['num_layers'],
        'd_ff': model._config['d_ff'],
    }

    # Guardar traza
    with open(trace_path, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Datos de traza guardados en: {trace_path}")

    # =====================================================================
    # RESUMEN FINAL
    # =====================================================================
    print(f"\n{'=' * 60}")
    print(f"🎉 ¡Entrenamiento completado!")
    print(f"{'=' * 60}")
    print(f"  📁 Modelo guardado en: {save_path}")
    print(f"  📊 Datos de traza en: {trace_path}")
    print(f"  📉 Mejor loss: {best_loss:.4f}")
    print(f"  📝 Perplexity final: {training_history[-1]['perplexity']:.1f}")
    print(f"  🔢 Parámetros: {model.count_parameters():,}")
    print(f"\n  Para visualizar, abre visualize_transformer.html en tu navegador.")

    return model, tokenizer, training_history


# =============================================================================
# FUNCIÓN DE DEMOSTRACIÓN INTERACTIVA
# =============================================================================

def interactive_demo(model_path: str = 'modelo_entrenado.pt'):
    """
    Demo interactiva: escribe un prompt y el modelo genera texto.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MiniTransformer.load(model_path, device=device)

    # Reconstruir tokenizer
    tokenizer = CharTokenizer()
    tokenizer.token_to_id = model.token_to_id
    tokenizer.id_to_token = model.id_to_token
    tokenizer.vocab_size = model.vocab_size

    print("\n" + "=" * 60)
    print("🐉 Demo Interactiva del Mini Transformer")
    print("=" * 60)
    print("Escribe un texto inicial y el modelo lo continuará.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        prompt = input("📝 Tu prompt: ")
        if prompt.lower() in ('salir', 'exit', 'quit'):
            print("👋 ¡Hasta luego!")
            break

        if not prompt:
            prompt = "El dragón "

        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long
        ).to(device)

        if prompt_ids.size(1) == 0:
            print("⚠️  Algunos caracteres no están en el vocabulario. Intenta otro texto.")
            continue

        generated = model.generate(
            prompt_ids,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
        text = tokenizer.decode(generated[0].tolist())
        print(f"\n🐉 Generado:\n{text}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        interactive_demo()
    else:
        train_model()
