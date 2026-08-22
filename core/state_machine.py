"""
Máquina de estados da aplicação.

Antes: `estado_app = "MENU"` como string solta + várias outras variáveis
globais (contador_fixacao, opcao_focada, etc.) soltas no módulo principal.

Agora: um Enum para os estados (evita erros de digitação tipo "MENU " com
espaço, ou "menu" minúsculo) e uma classe `AppState` que agrupa todo o
estado mutável em um único objeto, fácil de passar entre funções e de
testar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class EstadoApp(Enum):
    MENU = auto()
    SELECIONAR_PERFIL = auto()
    CALIB_PREPARAR = auto()
    CALIB_COLETANDO = auto()
    CALIB_CONCLUIDA = auto()
    ATIVO = auto()


@dataclass
class AppState:
    """Agrupa todo o estado mutável do app em um único objeto."""

    estado: EstadoApp = EstadoApp.MENU
    calib_fase: str = "ESQUERDA"
    coletas_calibracao: list[float] = field(default_factory=list)

    contador_fixacao: int = 0
    opcao_focada: str | None = None
    opcao_confirmada: str | None = None
    tempo_selecionado: float = 0.0

    # Confirmação alternativa por piscada
    contador_piscada: int = 0

    # Robustez: quantos frames seguidos sem detectar rosto
    contador_sem_rosto: int = 0

    # Perfil / calibração ativa
    perfil_ativo: str | None = None
    media_esquerda_calibrada: float = 0.35
    media_direita_calibrada: float = 0.65
    calibracao_ja_feita: bool = False

    def resetar_selecao(self) -> None:
        self.opcao_focada = None
        self.opcao_confirmada = None
        self.contador_fixacao = 0
        self.contador_piscada = 0

    def resetar_calibracao(self) -> None:
        self.calib_fase = "ESQUERDA"
        self.coletas_calibracao = []

    def transicionar(self, novo_estado: EstadoApp) -> None:
        """Ponto único de transição -- facilita logar/depurar mudanças de estado."""
        self.estado = novo_estado
