"""
============================================================================
utils/exportar_visualizacion.py — Exportar datos para visualización HTML/JS
============================================================================

Genera los archivos JSON necesarios para la visualización interactiva.
Recopila datos del tokenizador, modelo, y sistema RAG para crear
una experiencia educativa completa en el navegador.
============================================================================
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional


class ExportadorVisualizacion:
    """
    Recopila y exporta datos de todos los componentes del sistema
    para la visualización HTML/JS interactiva.
    """

    def __init__(self, directorio_salida: str = "visualizacion"):
        self.directorio = directorio_salida
        os.makedirs(directorio_salida, exist_ok=True)

    def exportar_tokenizacion(self, tokenizador, codigos_ejemplo: List[str]) -> Dict:
        """
        Exporta datos de tokenización para visualización.

        Incluye:
        - Tokens con colores por categoría (keyword, tipo, operador, etc.)
        - Estadísticas del vocabulario
        - Comparación de tokenización C# vs texto genérico
        """
        resultados = []

        for codigo in codigos_ejemplo:
            stats = tokenizador.obtener_estadisticas_tokenizacion(codigo)
            tokens_coloreados = []

            for token in stats["tokens"]:
                categoria = self._clasificar_token(token, tokenizador)
                token_id = tokenizador.token_a_id.get(token, -1)
                tokens_coloreados.append({
                    "texto": token,
                    "id": token_id,
                    "categoria": categoria,
                    "color": self._color_por_categoria(categoria),
                })

            resultados.append({
                "codigo_original": codigo,
                "tokens": tokens_coloreados,
                "estadisticas": {
                    "total_tokens": stats["total_tokens"],
                    "tokens_unicos": stats["tokens_unicos"],
                    "ratio_compresion": round(stats["ratio_compresion"], 2),
                    "por_categoria": {
                        k: len(v) for k, v in stats["clasificacion"].items()
                    },
                },
            })

        datos = {
            "ejemplos": resultados,
            "vocabulario_info": {
                "total_tokens": tokenizador.tam_vocabulario,
                "tokens_especiales": [
                    tokenizador.PAD_TOKEN, tokenizador.UNK_TOKEN,
                    tokenizador.BOS_TOKEN, tokenizador.EOS_TOKEN,
                    tokenizador.FIM_PREFIX, tokenizador.FIM_SUFFIX,
                    tokenizador.FIM_MIDDLE,
                ],
                "num_keywords_csharp": len(tokenizador.CSHARP_KEYWORDS),
                "num_operadores": len(tokenizador.OPERADORES_COMPUESTOS),
            },
        }

        self._guardar_json("datos_tokenizacion.json", datos)
        return datos

    def exportar_atencion(self, modelo, tokenizador, codigos_ejemplo: List[str]) -> Dict:
        """
        Exporta pesos de atención para visualización de attention maps.

        Los pesos de atención muestran a qué tokens "mira" el modelo
        en cada posición. Esto revela qué relaciones aprendió:
        - ¿Presta atención a las llaves de apertura al generar "}"?
        - ¿Presta atención al tipo al generar valores?
        """
        import torch

        modelo.eval()
        # Activar guardado de atención
        config_original = modelo.config.guardar_atencion
        modelo.config.guardar_atencion = True
        for capa in modelo.capas:
            capa.atencion.guardar_atencion = True

        resultados = []

        for codigo in codigos_ejemplo:
            ids = tokenizador.codificar(codigo, agregar_especiales=True)
            ids_tensor = torch.tensor([ids[:64]])  # Limitar para visualización

            with torch.no_grad():
                _ = modelo(ids_tensor)

            pesos = modelo.obtener_pesos_atencion()
            tokens = [tokenizador.id_a_token.get(i, "<?>") for i in ids[:64]]

            atenciones_por_capa = []
            for capa_idx, peso in enumerate(pesos):
                if peso is not None:
                    # Promediar sobre las cabezas para simplificar visualización
                    atencion_promedio = peso[0].mean(dim=0).cpu().numpy()
                    # Convertir a lista para JSON (solo primeras 32 posiciones)
                    max_dim = min(32, atencion_promedio.shape[0])
                    atenciones_por_capa.append({
                        "capa": capa_idx,
                        "pesos": atencion_promedio[:max_dim, :max_dim].tolist(),
                        "num_cabezas": peso.shape[1],
                    })

            resultados.append({
                "codigo": codigo[:200],
                "tokens": tokens[:32],
                "atenciones": atenciones_por_capa,
            })

        # Restaurar configuración
        modelo.config.guardar_atencion = config_original
        for capa in modelo.capas:
            capa.atencion.guardar_atencion = config_original

        datos = {"ejemplos": resultados}
        self._guardar_json("datos_atencion.json", datos)
        return datos

    def exportar_generacion(self, entrenador, ejemplos_prompt: List[str]) -> Dict:
        """
        Exporta datos de generación token por token.

        Incluye para cada paso:
        - Token generado
        - Probabilidad del token elegido
        - Top-10 candidatos alternativos
        """
        resultados = []

        for prompt in ejemplos_prompt:
            resultado = entrenador.completar_codigo(
                prompt, max_tokens=50, temperatura=0.8
            )

            # Convertir info de pasos para JSON
            pasos_json = []
            for paso in resultado["info_pasos"]:
                token_texto = entrenador.tokenizador.id_a_token.get(
                    paso["token_id"], "<UNK>"
                )
                candidatos = []
                for cand in paso["top_candidatos"][:5]:
                    cand_texto = entrenador.tokenizador.id_a_token.get(
                        cand["id"], "<UNK>"
                    )
                    candidatos.append({
                        "token": cand_texto,
                        "probabilidad": round(cand["prob"], 4),
                    })

                pasos_json.append({
                    "paso": paso["paso"],
                    "token": token_texto,
                    "probabilidad": round(paso["probabilidad"], 4),
                    "candidatos": candidatos,
                })

            resultados.append({
                "prompt": prompt,
                "codigo_generado": resultado["codigo_generado"],
                "tokens_generados": resultado["tokens_generados"],
                "pasos": pasos_json,
            })

        datos = {"ejemplos": resultados}
        self._guardar_json("datos_generacion.json", datos)
        return datos

    def exportar_rag(self, sistema_rag, queries_ejemplo: List[str]) -> Dict:
        """
        Exporta datos del sistema RAG para visualización del flujo completo.

        Incluye:
        - Query → embedding → búsqueda → resultados
        - Similitudes entre query y todos los documentos (heat map)
        - Contexto generado para cada query
        """
        resultados = []

        for query in queries_ejemplo:
            datos_vis = sistema_rag.exportar_datos_visualizacion(query)

            resultados.append({
                "query": query,
                "resultados": datos_vis["resultados"],
                "todos_documentos": datos_vis["todos_documentos"],
                "tiempo_busqueda_ms": round(datos_vis["tiempo_busqueda_ms"], 2),
                "contexto": datos_vis["contexto_generado"][:500],
                "embedding_query_preview": datos_vis["query_embedding_preview"][:10],
            })

        datos = {
            "ejemplos": resultados,
            "estadisticas": datos_vis.get("estadisticas_base", {}),
        }

        self._guardar_json("datos_rag.json", datos)
        return datos

    def exportar_comparacion_rag(self, comparaciones: List[Dict]) -> Dict:
        """Exporta datos de comparación con/sin RAG."""
        datos = {"comparaciones": comparaciones}
        self._guardar_json("datos_comparacion.json", datos)
        return datos

    def exportar_historial_entrenamiento(self, historial: List[Dict]) -> Dict:
        """Exporta historial de entrenamiento para gráficas."""
        datos = {"historial": historial}
        self._guardar_json("datos_entrenamiento.json", datos)
        return datos

    def _clasificar_token(self, token: str, tokenizador) -> str:
        """Clasifica un token por tipo para colorear."""
        if token in tokenizador.CSHARP_KEYWORDS:
            return "keyword"
        elif token in (tokenizador.NEWLINE_TOKEN, tokenizador.INDENT_TOKEN, tokenizador.DEDENT_TOKEN):
            return "whitespace"
        elif token in (tokenizador.FIM_PREFIX, tokenizador.FIM_SUFFIX, tokenizador.FIM_MIDDLE):
            return "fim"
        elif token in tokenizador.OPERADORES_SIMPLES or token in tokenizador.OPERADORES_COMPUESTOS:
            return "operator"
        elif token.startswith('"') or token.startswith("$\""):
            return "string"
        elif token.startswith('//'):
            return "comment"
        elif token[0:1].isdigit():
            return "number"
        elif token[0:1].isupper():
            return "type_or_class"
        else:
            return "identifier"

    def _color_por_categoria(self, categoria: str) -> str:
        """Retorna color CSS para cada categoría de token."""
        colores = {
            "keyword": "#569CD6",       # Azul (como VS Code)
            "type_or_class": "#4EC9B0", # Verde azulado
            "identifier": "#9CDCFE",    # Azul claro
            "operator": "#D4D4D4",      # Gris claro
            "string": "#CE9178",        # Naranja
            "number": "#B5CEA8",        # Verde claro
            "comment": "#6A9955",       # Verde
            "whitespace": "#333333",    # Gris oscuro
            "fim": "#C586C0",           # Púrpura
        }
        return colores.get(categoria, "#D4D4D4")

    def _guardar_json(self, nombre: str, datos: Dict):
        """Guarda datos como JSON."""
        ruta = os.path.join(self.directorio, nombre)
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(datos, f, ensure_ascii=False, indent=2, default=str)
        print(f"📄 Exportado: {ruta}")
