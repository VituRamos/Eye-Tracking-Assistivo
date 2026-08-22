"""
Persistência de calibração por perfil de usuário.

Antes, a calibração (`media_esquerda_calibrada`, `media_direita_calibrada`)
vivia só na memória e se perdia ao fechar o programa -- toda sessão
exigia recalibrar. Agora cada perfil (ex: um paciente/usuário) tem um
arquivo JSON próprio em `perfis/`, com data da última calibração e uma
nota simples de qualidade.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger("eyetracking.calibration")


class DadosCalibracao(TypedDict):
    nome: str
    media_esquerda: float
    media_direita: float
    data_calibracao: str
    qualidade: str


class GerenciadorCalibracao:
    def __init__(self, pasta: str | Path = "perfis", diferenca_minima_boa: float = 0.15):
        self.pasta = Path(pasta)
        self.pasta.mkdir(exist_ok=True, parents=True)
        self.diferenca_minima_boa = diferenca_minima_boa

    def _caminho(self, nome_perfil: str) -> Path:
        # Validação estrita: rejeita (em vez de apenas filtrar) qualquer
        # caractere fora do permitido. Só remover caracteres inválidos
        # (como fazia a versão anterior) permitiria que "../../etc/passwd"
        # virasse "etcpasswd" silenciosamente -- aqui isso é bloqueado.
        nome = nome_perfil.strip()
        if not nome or not all(c.isalnum() or c in ("_", "-") for c in nome):
            raise ValueError(
                f"Nome de perfil inválido: '{nome_perfil}'. "
                "Use apenas letras, números, '_' ou '-'."
            )
        return self.pasta / f"{nome}.json"

    def salvar(self, nome_perfil: str, media_esq: float, media_dir: float) -> DadosCalibracao:
        qualidade = "boa" if abs(media_dir - media_esq) >= self.diferenca_minima_boa else "fraca"
        dados: DadosCalibracao = {
            "nome": nome_perfil,
            "media_esquerda": round(media_esq, 4),
            "media_direita": round(media_dir, 4),
            "data_calibracao": datetime.now().isoformat(timespec="seconds"),
            "qualidade": qualidade,
        }
        caminho = self._caminho(nome_perfil)
        caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Calibração salva para perfil '%s' (qualidade=%s).", nome_perfil, qualidade)
        return dados

    def carregar(self, nome_perfil: str) -> DadosCalibracao | None:
        caminho = self._caminho(nome_perfil)
        if not caminho.exists():
            return None
        try:
            return json.loads(caminho.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Falha ao ler calibração de '%s': %s", nome_perfil, e)
            return None

    def listar_perfis(self) -> list[str]:
        return sorted(p.stem for p in self.pasta.glob("*.json"))

    def calibracao_esta_vencida(self, dados: DadosCalibracao, validade_dias: int = 7) -> bool:
        """Indica se vale sugerir recalibração em vez de reusar automaticamente."""
        try:
            data = datetime.fromisoformat(dados["data_calibracao"])
        except (KeyError, ValueError):
            return True
        return datetime.now() - data > timedelta(days=validade_dias)
