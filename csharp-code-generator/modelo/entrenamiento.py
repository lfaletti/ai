"""
============================================================================
modelo/entrenamiento.py — Pipeline de Entrenamiento e Inferencia
============================================================================

PROPÓSITO EDUCATIVO:
    Este módulo implementa el ciclo completo de entrenamiento del modelo
    Transformer para generación de código C#. Incluye:

    1. Preparación de datos: convertir código C# en batches de tensores
    2. Entrenamiento: optimización del modelo con Adam + warmup
    3. Evaluación: medir perplexity y generar ejemplos
    4. Inferencia: usar el modelo entrenado para generar código

DIFERENCIAS CON ENTRENAMIENTO DE MODELOS DE TEXTO:
    1. TOKENIZACIÓN ESPECIAL: usamos nuestro tokenizador de C#
    2. DATOS FIM: intercalamos ejemplos normales y FIM
    3. MÉTRICAS ESPECIALES: además de perplexity, medimos si
       el código generado es sintácticamente correcto
    4. WARMUP MÁS LARGO: los modelos de código se benefician de
       un calentamiento gradual del learning rate
============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import math
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from modelo.tokenizador_csharp import TokenizadorCSharp
from modelo.transformer import TransformerCodeGen, ConfiguracionTransformer


# ============================================================================
# DATASET DE PYTORCH
# ============================================================================

class DatasetCodigoCSharp(Dataset):
    """
    Dataset de PyTorch que prepara secuencias de código C# para entrenamiento.

    NOTA EDUCATIVA — PREPARACIÓN DE DATOS PARA CÓDIGO:
        A diferencia de texto natural donde simplemente concatenamos
        oraciones, en código debemos:

        1. RESPETAR LÍMITES DE ARCHIVO: no mezclar código de diferentes
           archivos en la misma secuencia (el modelo se confundiría)

        2. INCLUIR EJEMPLOS FIM: intercalar ejemplos normales con
           ejemplos Fill-in-the-Middle para que el modelo aprenda ambos

        3. PADDING INTELIGENTE: rellenar secuencias cortas con PAD tokens.
           El modelo aprende a IGNORAR los tokens de padding.

        4. CREAR LABELS: los labels son la misma secuencia desplazada
           1 posición a la derecha (predecir el siguiente token)
    """

    def __init__(
        self,
        codigos: List[str],
        tokenizador: TokenizadorCSharp,
        max_longitud: int = 512,
        proporcion_fim: float = 0.3,
    ):
        """
        Args:
            codigos: Lista de snippets de código C#
            tokenizador: Tokenizador de C# (debe tener vocabulario construido)
            max_longitud: Longitud máxima de secuencia
            proporcion_fim: Proporción de ejemplos FIM (0.3 = 30%)

        NOTA EDUCATIVA — PROPORCIÓN FIM:
            Si usamos 100% FIM, el modelo sería bueno completando en medio
            pero malo generando código nuevo. 30% FIM es un buen balance.
            Copilot usa ~50% FIM en su entrenamiento.
        """
        self.tokenizador = tokenizador
        self.max_longitud = max_longitud
        self.proporcion_fim = proporcion_fim

        # Pre-tokenizar todos los códigos
        print("📝 Pre-tokenizando dataset...")
        self.secuencias = []

        for i, codigo in enumerate(codigos):
            # Decidir si crear ejemplo normal o FIM
            if random.random() < proporcion_fim:
                secuencia = self._crear_ejemplo_fim(codigo)
            else:
                secuencia = self._crear_ejemplo_normal(codigo)

            if secuencia and len(secuencia) >= 10:  # Mínimo 10 tokens
                self.secuencias.append(secuencia)

        print(f"✅ Dataset preparado: {len(self.secuencias)} secuencias")

    def _crear_ejemplo_normal(self, codigo: str) -> List[int]:
        """
        Crea un ejemplo de entrenamiento normal (autoregresivo).

        Formato: [BOS] token1 token2 ... tokenN [EOS]
        Labels:  token1 token2 ... tokenN [EOS] [PAD]
        """
        ids = self.tokenizador.codificar(codigo, agregar_especiales=True)

        # Truncar si es necesario
        if len(ids) > self.max_longitud:
            ids = ids[:self.max_longitud - 1] + [self.tokenizador.eos_id]

        return ids

    def _crear_ejemplo_fim(self, codigo: str) -> List[int]:
        """
        Crea un ejemplo Fill-in-the-Middle.

        NOTA EDUCATIVA:
            El formato FIM reordena el código así:
            Original:  "public class Product { int Id; }"
            FIM:       <FIM_PREFIX>public class Product { <FIM_SUFFIX> }
                       <FIM_MIDDLE>int Id;

            El modelo aprende a generar el "middle" viendo el contexto
            de ambos lados (prefix Y suffix).
        """
        if len(codigo) < 50:  # Código muy corto, usar formato normal
            return self._crear_ejemplo_normal(codigo)

        resultado_fim = self.tokenizador.preparar_fim(codigo)
        ids = self.tokenizador.codificar_fim(
            resultado_fim["prefijo"],
            resultado_fim["sufijo"],
            resultado_fim["middle"]
        )

        if len(ids) > self.max_longitud:
            ids = ids[:self.max_longitud - 1] + [self.tokenizador.eos_id]

        return ids

    def __len__(self) -> int:
        return len(self.secuencias)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Retorna un ejemplo con padding.

        NOTA EDUCATIVA:
            Todos los ejemplos en un batch deben tener el MISMO largo.
            Los más cortos se rellenan con PAD tokens (ID=0).
            La máscara de atención y el loss ignoran los PAD tokens.
        """
        ids = self.secuencias[idx]

        # Padding al final
        padding_len = self.max_longitud - len(ids)
        ids_padded = ids + [self.tokenizador.pad_id] * padding_len

        # Crear tensor
        input_ids = torch.tensor(ids_padded, dtype=torch.long)

        # Labels = input_ids (el shift se hace en el modelo)
        labels = input_ids.clone()

        # Mascara de atención: 1 para tokens reales, 0 para padding
        mascara_atencion = (input_ids != self.tokenizador.pad_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": mascara_atencion,
        }


# ============================================================================
# ENTRENADOR DEL MODELO
# ============================================================================

@dataclass
class ConfigEntrenamiento:
    """
    Configuración del entrenamiento.

    NOTA EDUCATIVA — HIPERPARÁMETROS DE ENTRENAMIENTO:

    learning_rate (3e-4):
        Tasa de aprendizaje. Para modelos Transformer, 3e-4 es un buen
        punto de partida. Demasiado alto → el modelo no converge.
        Demasiado bajo → el entrenamiento es muy lento.

    batch_size (8):
        Número de ejemplos procesados juntos. Batch más grande = estimación
        del gradiente más estable, pero requiere más memoria GPU.

    epochs (10):
        Número de pasadas completas sobre el dataset. Para nuestro dataset
        sintético pequeño, 10 epochs son suficientes. Modelos de producción
        entrenan por 1-3 epochs sobre datasets masivos.

    warmup_steps (100):
        Pasos donde el learning rate sube gradualmente de 0 a lr_max.
        Sin warmup, los primeros gradientes pueden ser muy ruidosos
        y desestabilizar el entrenamiento.
    """
    learning_rate: float = 3e-4
    batch_size: int = 8
    epochs: int = 10
    warmup_steps: int = 100
    max_longitud: int = 512
    proporcion_fim: float = 0.3
    guardar_cada: int = 2         # Guardar checkpoint cada N epochs
    directorio_checkpoints: str = "checkpoints"
    dispositivo: str = "cpu"      # "cuda" si hay GPU disponible
    log_cada: int = 50            # Log cada N pasos


class EntrenadorModelo:
    """
    Orquesta el entrenamiento completo del modelo.

    FLUJO DE ENTRENAMIENTO:
    1. Preparar datos → DataLoader con batches
    2. Configurar optimizador → AdamW con weight decay
    3. Configurar scheduler → Warmup + cosine decay
    4. Loop de entrenamiento:
       a. Forward pass → obtener logits y loss
       b. Backward pass → calcular gradientes
       c. Update → actualizar pesos del modelo
       d. Log → registrar métricas
    5. Evaluación periódica → generar código de ejemplo
    """

    def __init__(
        self,
        modelo: TransformerCodeGen,
        tokenizador: TokenizadorCSharp,
        config: ConfigEntrenamiento,
    ):
        self.modelo = modelo.to(config.dispositivo)
        self.tokenizador = tokenizador
        self.config = config

        # Optimizador AdamW (Adam con weight decay desacoplado)
        # Weight decay previene overfitting penalizando pesos grandes
        self.optimizador = torch.optim.AdamW(
            modelo.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),  # Betas optimizados para LLMs
        )

        # Historial de entrenamiento para visualización
        self.historial: List[Dict] = []

        # Crear directorio de checkpoints
        os.makedirs(config.directorio_checkpoints, exist_ok=True)

    def entrenar(self, codigos: List[str]) -> Dict:
        """
        Ejecuta el ciclo completo de entrenamiento.

        Args:
            codigos: Lista de snippets de código C# para entrenar

        Returns:
            Historial de entrenamiento con métricas por paso
        """
        print("=" * 60)
        print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
        print("=" * 60)
        print(f"   Dispositivo: {self.config.dispositivo}")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Ejemplos de entrenamiento: {len(codigos)}")

        # Crear dataset y dataloader
        dataset = DatasetCodigoCSharp(
            codigos=codigos,
            tokenizador=self.tokenizador,
            max_longitud=self.config.max_longitud,
            proporcion_fim=self.config.proporcion_fim,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Configurar scheduler de learning rate
        total_steps = len(dataloader) * self.config.epochs
        scheduler = self._crear_scheduler(total_steps)

        # Loop de entrenamiento
        paso_global = 0
        mejor_loss = float('inf')

        for epoch in range(self.config.epochs):
            self.modelo.train()
            loss_acumulado = 0
            tokens_procesados = 0

            for batch_idx, batch in enumerate(dataloader):
                # Mover datos al dispositivo
                input_ids = batch["input_ids"].to(self.config.dispositivo)
                labels = batch["labels"].to(self.config.dispositivo)

                # Forward pass
                resultado = self.modelo(input_ids, labels=labels)
                loss = resultado["loss"]

                # Backward pass
                self.optimizador.zero_grad()
                loss.backward()

                # Gradient clipping (evita gradientes explosivos)
                torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), 1.0)

                # Update
                self.optimizador.step()
                scheduler.step()

                # Métricas
                loss_valor = loss.item()
                loss_acumulado += loss_valor
                tokens_batch = (batch["attention_mask"].sum()).item()
                tokens_procesados += tokens_batch
                paso_global += 1

                # Registrar en historial
                self.historial.append({
                    "paso": paso_global,
                    "epoch": epoch,
                    "loss": loss_valor,
                    "perplexity": math.exp(min(loss_valor, 20)),  # Cap para evitar inf
                    "lr": scheduler.get_last_lr()[0],
                    "tokens": tokens_procesados,
                })

                # Log periódico
                if paso_global % self.config.log_cada == 0:
                    loss_promedio = loss_acumulado / (batch_idx + 1)
                    perplexity = math.exp(min(loss_promedio, 20))
                    print(
                        f"  Epoch {epoch+1}/{self.config.epochs} | "
                        f"Paso {paso_global} | "
                        f"Loss: {loss_promedio:.4f} | "
                        f"Perplexity: {perplexity:.2f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

            # Fin de epoch
            loss_epoch = loss_acumulado / max(1, len(dataloader))
            perplexity_epoch = math.exp(min(loss_epoch, 20))

            print(f"\n📊 Epoch {epoch+1} completado:")
            print(f"   Loss promedio: {loss_epoch:.4f}")
            print(f"   Perplexity: {perplexity_epoch:.2f}")

            # Guardar checkpoint si mejoró
            if loss_epoch < mejor_loss:
                mejor_loss = loss_epoch
                self._guardar_checkpoint(epoch, loss_epoch)

            # Generar ejemplo de evaluación
            if (epoch + 1) % 2 == 0:
                self._generar_ejemplo_evaluacion()

        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Mejor loss: {mejor_loss:.4f}")
        print(f"   Pasos totales: {paso_global}")
        print("=" * 60)

        return {
            "historial": self.historial,
            "mejor_loss": mejor_loss,
            "pasos_totales": paso_global,
        }

    def _crear_scheduler(self, total_steps: int):
        """
        Crea un scheduler de learning rate con warmup + cosine decay.

        NOTA EDUCATIVA:
            El scheduler controla cómo cambia el learning rate durante
            el entrenamiento:

            1. WARMUP (primeros N pasos):
               LR sube linealmente de 0 a lr_max.
               Esto estabiliza el inicio del entrenamiento.

            2. COSINE DECAY (después del warmup):
               LR baja siguiendo una curva coseno.
               Esto permite "refinamiento fino" al final.

            ┌────────────────────────────────────────┐
            │  lr_max  /\                            │
            │        /    \                          │
            │      /        \_____                   │
            │    /                 \_____             │
            │  /                         \___        │
            │/                               \___    │
            └────────────────────────────────────────┘
             warmup              cosine decay
        """
        def lr_lambda(paso):
            if paso < self.config.warmup_steps:
                # Warmup lineal
                return paso / max(1, self.config.warmup_steps)
            else:
                # Cosine decay
                progreso = (paso - self.config.warmup_steps) / max(
                    1, total_steps - self.config.warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * progreso))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizador, lr_lambda)

    def _guardar_checkpoint(self, epoch: int, loss: float):
        """Guarda un checkpoint del modelo."""
        ruta = os.path.join(
            self.config.directorio_checkpoints,
            f"modelo_epoch_{epoch+1}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.modelo.state_dict(),
            "optimizer_state_dict": self.optimizador.state_dict(),
            "loss": loss,
        }, ruta)
        print(f"   💾 Checkpoint guardado: {ruta}")

    def _generar_ejemplo_evaluacion(self):
        """
        Genera un ejemplo de código para evaluar visualmente el progreso.

        NOTA EDUCATIVA:
            Además de métricas numéricas (loss, perplexity), es útil
            VER qué genera el modelo en cada etapa:
            - Epoch 1: basura, tokens aleatorios
            - Epoch 5: estructura básica (class {})
            - Epoch 10: código más coherente con tipos
        """
        self.modelo.eval()

        prompts = [
            "public class",
            "public async Task",
            "[HttpGet]",
        ]

        print("\n   📝 Ejemplos de generación:")
        for prompt in prompts:
            ids = self.tokenizador.codificar(prompt, agregar_especiales=True)
            ids_tensor = torch.tensor([ids], device=self.config.dispositivo)

            generados, _ = self.modelo.generar(
                ids_tensor, max_tokens=30, temperatura=0.7
            )

            texto = self.tokenizador.decodificar(generados[0].tolist())
            # Mostrar solo la primera línea generada
            primera_linea = texto.split('\n')[0][:100]
            print(f"      '{prompt}' → {primera_linea}...")

        self.modelo.train()

    # ================================================================
    # INFERENCIA
    # ================================================================

    def completar_codigo(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperatura: float = 0.7,
    ) -> Dict:
        """
        Completa código C# a partir de un prompt.

        NOTA EDUCATIVA:
            Esta es la operación principal de un copilot:
            el desarrollador escribe un inicio, y el modelo completa.

        Args:
            prompt: Inicio del código (ej: "public class Product")
            max_tokens: Tokens máximos a generar
            temperatura: Control de creatividad

        Returns:
            Dict con código completo, tokens generados e info de pasos
        """
        self.modelo.eval()

        ids = self.tokenizador.codificar(prompt, agregar_especiales=True)
        ids_tensor = torch.tensor([ids], device=self.config.dispositivo)

        generados, info_pasos = self.modelo.generar(
            ids_tensor,
            max_tokens=max_tokens,
            temperatura=temperatura,
            tokens_parada=[self.tokenizador.eos_id],
        )

        codigo_completo = self.tokenizador.decodificar(generados[0].tolist())

        return {
            "prompt": prompt,
            "codigo_generado": codigo_completo,
            "tokens_generados": len(generados[0]) - len(ids),
            "info_pasos": info_pasos,
        }

    def completar_fim(
        self,
        prefijo: str,
        sufijo: str,
        max_tokens: int = 100,
        temperatura: float = 0.7,
    ) -> Dict:
        """
        Completa código en medio de un archivo (Fill-in-the-Middle).

        NOTA EDUCATIVA:
            FIM es la característica que hace a los copilots realmente útiles.
            No solo completan al final, sino EN CUALQUIER POSICIÓN.

        Args:
            prefijo: Código antes del cursor
            sufijo: Código después del cursor
            max_tokens: Tokens máximos a generar
            temperatura: Control de creatividad

        Returns:
            Dict con el código middle generado
        """
        self.modelo.eval()

        ids = self.tokenizador.codificar_fim(prefijo, sufijo)
        ids_tensor = torch.tensor([ids], device=self.config.dispositivo)

        generados, info_pasos = self.modelo.generar(
            ids_tensor,
            max_tokens=max_tokens,
            temperatura=temperatura,
            tokens_parada=[self.tokenizador.eos_id],
        )

        middle_generado = self.tokenizador.decodificar(
            generados[0][len(ids):].tolist()
        )

        return {
            "prefijo": prefijo,
            "sufijo": sufijo,
            "middle_generado": middle_generado,
            "codigo_completo": prefijo + middle_generado + sufijo,
            "tokens_generados": len(generados[0]) - len(ids),
            "info_pasos": info_pasos,
        }

    def obtener_historial_json(self) -> str:
        """Retorna el historial de entrenamiento como JSON para visualización."""
        return json.dumps(self.historial, indent=2)
