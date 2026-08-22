"""
Testes unitários das funções puras de core/gaze_engine.py.

Como essas funções não dependem de câmera nem de MediaPipe rodando de
verdade, usamos objetos falsos (FakeLandmark) só com os atributos .x/.y
que as funções esperam. Isso permite testar toda a lógica de cálculo
sem precisar de uma webcam nem de um rosto real.

Rode com: pytest tests/
"""
from dataclasses import dataclass

import numpy as np
import pytest

from core.gaze_engine import (
    SuavizadorEMA,
    calcular_ear,
    calcular_ratio_iris_x,
    classificar_lado,
    compensar_ratio_por_yaw,
    extrair_yaw_da_matriz,
    olho_fechado,
)


@dataclass
class FakeLandmark:
    x: float
    y: float


def montar_landmarks_olho(x_canto_esq, x_canto_dir, x_iris, y=0.5):
    """
    Monta um dicionário simulando os landmarks do MediaPipe para um olho,
    usando os mesmos índices reais (33/133 do olho direito, por exemplo),
    mas preenchendo só o que os testes precisam.
    """
    indices_iris = [469, 470, 471, 472]
    landmarks = {33: FakeLandmark(x_canto_esq, y), 133: FakeLandmark(x_canto_dir, y)}
    for i in indices_iris:
        landmarks[i] = FakeLandmark(x_iris, y)
    return landmarks


# =====================================================================
# classificar_lado
# =====================================================================
class TestClassificarLado:
    def test_classifica_esquerda_quando_ratio_baixo(self):
        assert classificar_lado(0.2, media_esq=0.3, media_dir=0.7) == "Esquerda"

    def test_classifica_direita_quando_ratio_alto(self):
        assert classificar_lado(0.8, media_esq=0.3, media_dir=0.7) == "Direita"

    def test_zona_morta_retorna_centro(self):
        # Bem no meio do intervalo calibrado -> deve cair na zona morta
        assert classificar_lado(0.5, media_esq=0.3, media_dir=0.7) == "Centro"

    def test_funciona_com_calibracao_invertida(self):
        # Se por algum motivo o SIM ficou calibrado com ratio maior que o NAO,
        # a classificação ainda deve respeitar qual lado é qual.
        assert classificar_lado(0.8, media_esq=0.7, media_dir=0.3) == "Esquerda"


# =====================================================================
# calcular_ratio_iris_x
# =====================================================================
class TestCalcularRatioIrisX:
    def test_iris_no_centro_do_olho(self):
        largura, altura = 640, 480
        landmarks = montar_landmarks_olho(
            x_canto_esq=0.3, x_canto_dir=0.4, x_iris=0.35,
        )
        ratio = calcular_ratio_iris_x(landmarks, (33, 133), [469, 470, 471, 472], largura, altura)
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_ratio_fica_limitado_entre_0_e_1(self):
        largura, altura = 640, 480
        # íris "fora" do olho (situação anômala) não deve estourar os limites
        landmarks = montar_landmarks_olho(x_canto_esq=0.3, x_canto_dir=0.4, x_iris=0.9)
        ratio = calcular_ratio_iris_x(landmarks, (33, 133), [469, 470, 471, 472], largura, altura)
        assert 0.0 <= ratio <= 1.0


# =====================================================================
# SuavizadorEMA
# =====================================================================
class TestSuavizadorEMA:
    def test_primeiro_valor_e_usado_diretamente(self):
        suavizador = SuavizadorEMA(alpha=0.3)
        assert suavizador.atualizar(0.5) == pytest.approx(0.5)

    def test_converge_para_valor_constante(self):
        suavizador = SuavizadorEMA(alpha=0.5)
        for _ in range(50):
            valor = suavizador.atualizar(0.8)
        assert valor == pytest.approx(0.8, abs=1e-6)

    def test_reset_limpa_estado(self):
        suavizador = SuavizadorEMA(alpha=0.5)
        suavizador.atualizar(0.9)
        suavizador.reset()
        assert suavizador.valor is None

    def test_alpha_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            SuavizadorEMA(alpha=0.0)
        with pytest.raises(ValueError):
            SuavizadorEMA(alpha=1.5)


# =====================================================================
# EAR / piscada
# =====================================================================
class TestEAR:
    def test_olho_bem_aberto_nao_conta_como_fechado(self):
        landmarks = {
            159: FakeLandmark(0.5, 0.40),
            145: FakeLandmark(0.5, 0.48),
            33: FakeLandmark(0.45, 0.44),
            133: FakeLandmark(0.55, 0.44),
        }
        ear = calcular_ear(landmarks, (159, 145), (33, 133), largura=640, altura=480)
        assert not olho_fechado(ear, limiar=0.15)

    def test_olho_quase_fechado_conta_como_fechado(self):
        landmarks = {
            159: FakeLandmark(0.5, 0.440),
            145: FakeLandmark(0.5, 0.442),
            33: FakeLandmark(0.45, 0.441),
            133: FakeLandmark(0.55, 0.441),
        }
        ear = calcular_ear(landmarks, (159, 145), (33, 133), largura=640, altura=480)
        assert olho_fechado(ear, limiar=0.15)


# =====================================================================
# Compensação de pose de cabeça
# =====================================================================
class TestCompensacaoPose:
    def test_matriz_identidade_da_yaw_zero(self):
        matriz = np.eye(4)
        yaw = extrair_yaw_da_matriz(matriz)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_compensar_ratio_com_yaw_zero_nao_altera_valor(self):
        ratio_ajustado = compensar_ratio_por_yaw(0.5, yaw_graus=0.0)
        assert ratio_ajustado == pytest.approx(0.5)

    def test_compensar_ratio_com_yaw_positivo_desloca_valor(self):
        ratio_ajustado = compensar_ratio_por_yaw(0.5, yaw_graus=20.0, sensibilidade=0.004)
        assert ratio_ajustado < 0.5
