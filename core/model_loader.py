"""
Download e verificação de integridade do modelo Face Landmarker.

Antes, o download acontecia silenciosamente com `print()` e sem checar
se o arquivo baixado estava íntegro. Agora valida checksum (se
configurado) e loga tudo via logging estruturado.
"""
from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger("eyetracking.model")


def _sha256_do_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def garantir_modelo_baixado(caminho: str, url: str, sha256_esperado: str = "") -> Path:
    """
    Baixa o modelo .task na primeira execução, se ainda não existir
    localmente. Se `sha256_esperado` for fornecido, valida a integridade
    do arquivo (baixado ou já existente) e re-baixa em caso de divergência.
    """
    caminho_path = Path(caminho)

    if caminho_path.exists() and sha256_esperado:
        checksum_atual = _sha256_do_arquivo(caminho_path)
        if checksum_atual != sha256_esperado:
            logger.warning(
                "Checksum do modelo local não confere (esperado=%s, obtido=%s). Baixando novamente.",
                sha256_esperado, checksum_atual,
            )
            caminho_path.unlink()

    if not caminho_path.exists():
        logger.info("Baixando modelo do MediaPipe Face Landmarker para '%s'...", caminho_path)
        try:
            urllib.request.urlretrieve(url, caminho_path)
        except Exception as e:
            raise RuntimeError(
                f"Falha ao baixar o modelo de '{url}'. Verifique sua conexão com a internet."
            ) from e
        logger.info("Download concluído.")

        if sha256_esperado:
            checksum_baixado = _sha256_do_arquivo(caminho_path)
            if checksum_baixado != sha256_esperado:
                caminho_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Checksum do modelo baixado não confere com o esperado. "
                    "O arquivo foi removido; tente novamente."
                )

    return caminho_path
