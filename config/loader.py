"""
Carregamento de configuração externa (YAML).

Mantém o sistema ajustável sem precisar editar código Python -- um
terapeuta ou cuidador pode alterar sensibilidade, resolução de câmera,
etc. só editando o settings.yaml.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("eyetracking.config")

CAMINHO_PADRAO = Path(__file__).parent / "settings.yaml"


def carregar_config(caminho: str | Path = CAMINHO_PADRAO) -> dict[str, Any]:
    """
    Carrega o settings.yaml. Se o arquivo não existir ou estiver corrompido,
    cai para valores padrão em código (para o app nunca travar por causa
    de config ausente/quebrada).
    """
    caminho = Path(caminho)
    if not caminho.exists():
        logger.warning("Arquivo de configuração '%s' não encontrado. Usando valores padrão.", caminho)
        return _config_padrao()

    try:
        conteudo = yaml.safe_load(caminho.read_text(encoding="utf-8"))
        if not conteudo:
            raise ValueError("Arquivo de configuração vazio.")
        return conteudo
    except Exception as e:
        logger.error("Falha ao ler '%s' (%s). Usando valores padrão.", caminho, e)
        return _config_padrao()


def _config_padrao() -> dict[str, Any]:
    """Valores de segurança caso o YAML não possa ser lido."""
    return {
        "deteccao": {
            "frames_para_selecionar": 15,
            "num_frames_calibracao": 90,
            "diferenca_minima_calibracao": 0.03,
            "alpha_ema": 0.35,
            "tamanho_historico_grafico": 150,
            "limite_frames_sem_rosto": 60,
            "piscada": {
                "habilitada": True,
                "limiar_ear": 0.15,
                "frames_minimos_piscada": 8,
            },
        },
        "camera": {"indice": 0, "largura": 1280, "altura": 720},
        "modelo": {
            "caminho": "face_landmarker.task",
            "url": (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            ),
            "sha256_esperado": "",
        },
        "perfis": {"pasta": "perfis", "validade_dias": 7},
        "logs": {"pasta": "logs", "nivel": "INFO"},
        "janela": {"titulo": "Eye-Tracking Assistivo V2.0 (MediaPipe)", "fullscreen": True},
    }
