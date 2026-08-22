"""
Logging estruturado.

Antes: `print()` espalhado no meio da lógica de calibração.
Agora: cada sessão gera um arquivo de log próprio em `logs/`, além de
imprimir no console. Isso serve tanto para depuração técnica quanto
para um terapeuta/cuidador revisar depois o que aconteceu numa sessão
(quantas seleções, qual a qualidade da calibração, erros de câmera etc.).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configurar_logging(pasta: str | Path = "logs", nivel: str = "INFO") -> logging.Logger:
    pasta = Path(pasta)
    pasta.mkdir(exist_ok=True, parents=True)

    nome_arquivo = pasta / f"sessao_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(nome_arquivo, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # garante reconfiguração mesmo se algo já chamou basicConfig antes
    )

    logger = logging.getLogger("eyetracking")
    logger.info("Sessão iniciada. Log salvo em: %s", nome_arquivo)
    return logger
