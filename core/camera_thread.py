"""
Captura de câmera em thread separada.

Antes, `cap.read()` era chamado direto no loop principal, junto com
detecção MediaPipe e desenho na mesma iteração -- se o processamento
atrasasse um frame, a UI travava junto. Agora a câmera roda em sua
própria thread, alimentando uma fila de tamanho 1 (sempre o frame mais
recente), e o loop principal só consome o que estiver disponível.
"""
from __future__ import annotations

import logging
import queue
import threading

import cv2

logger = logging.getLogger("eyetracking.camera")


class CameraIndisponivelError(RuntimeError):
    """Levantado quando a câmera não pode ser aberta."""


class CameraStream:
    def __init__(self, indice: int = 0, largura: int | None = None, altura: int | None = None):
        self.cap = cv2.VideoCapture(indice)
        if not self.cap.isOpened():
            raise CameraIndisponivelError(
                f"Não foi possível abrir a câmera de índice {indice}. "
                "Verifique se ela está conectada e não está em uso por outro programa."
            )

        if largura:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        if altura:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

        self.fila: queue.Queue = queue.Queue(maxsize=1)
        self._rodando = True
        self._thread = threading.Thread(target=self._capturar, daemon=True)
        self._thread.start()
        logger.info("Câmera %d iniciada.", indice)

    def _capturar(self) -> None:
        while self._rodando:
            sucesso, frame = self.cap.read()
            if not sucesso:
                continue
            # Mantém sempre só o frame mais recente -- evita acumular atraso
            if not self.fila.empty():
                try:
                    self.fila.get_nowait()
                except queue.Empty:
                    pass
            self.fila.put(frame)

    def ler(self, timeout: float = 1.0):
        """Bloqueia até um frame estar disponível (ou timeout)."""
        return self.fila.get(timeout=timeout)

    def parar(self) -> None:
        self._rodando = False
        self._thread.join(timeout=1.0)
        self.cap.release()
        logger.info("Câmera liberada.")
