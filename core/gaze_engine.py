"""
Motor de cálculo de gaze (rastreamento de olhar).

Todas as funções aqui são PURAS: recebem landmarks + números, devolvem
números. Nenhuma função desenha nada com OpenCV e nenhuma depende de
câmera -- por isso são fáceis de testar com dados falsos (ver
tests/test_gaze_engine.py).

A camada de desenho/debug (recorte da região do olho, retângulos, etc.)
fica em ui/renderer_cv2.py.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------
# Índices oficiais do MediaPipe Face Mesh (canônicos, não mudam entre rostos)
# ---------------------------------------------------------------------
OLHO_DIREITO_CANTOS = (33, 133)     # (canto externo, canto interno) do olho direito
OLHO_ESQUERDO_CANTOS = (362, 263)   # (canto interno, canto externo) do olho esquerdo
IRIS_DIREITA = [469, 470, 471, 472]
IRIS_ESQUERDA = [474, 475, 476, 477]

# Pontos de pálpebra superior/inferior, usados para EAR (piscada) e ratio_y
OLHO_DIREITO_VERTICAL = (159, 145)
OLHO_ESQUERDO_VERTICAL = (386, 374)


# =====================================================================
# Ratio horizontal (posição da íris dentro do olho)
# =====================================================================
def calcular_ratio_iris_x(landmarks, indices_cantos, indices_iris, largura, altura) -> float:
    """
    Retorna a posição horizontal da íris DENTRO da largura real daquele olho.
    0.0 = encostada no canto A, 1.0 = encostada no canto B.
    """
    p1 = landmarks[indices_cantos[0]]
    p2 = landmarks[indices_cantos[1]]
    x1_px, x2_px = p1.x * largura, p2.x * largura

    xs_iris = [landmarks[i].x * largura for i in indices_iris]
    iris_cx = float(np.mean(xs_iris))

    x_esq, x_dir = min(x1_px, x2_px), max(x1_px, x2_px)
    largura_olho = max(x_dir - x_esq, 1e-3)

    ratio = (iris_cx - x_esq) / largura_olho
    return min(max(ratio, 0.0), 1.0)


def calcular_ratio_iris_y(landmarks, indices_verticais, indices_iris, largura, altura) -> float:
    """
    Equivalente vertical do ratio_x. Abre caminho para uma grade 2D
    (não só SIM/NAO), útil para evoluir para um teclado por varredura.
    """
    p_sup = landmarks[indices_verticais[0]]
    p_inf = landmarks[indices_verticais[1]]
    y_sup, y_inf = p_sup.y * altura, p_inf.y * altura

    ys_iris = [landmarks[i].y * altura for i in indices_iris]
    iris_cy = float(np.mean(ys_iris))

    y_topo, y_base = min(y_sup, y_inf), max(y_sup, y_inf)
    altura_olho = max(y_base - y_topo, 1e-3)

    ratio = (iris_cy - y_topo) / altura_olho
    return min(max(ratio, 0.0), 1.0)


# =====================================================================
# Suavização -- EMA no lugar da média móvel simples
# =====================================================================
class SuavizadorEMA:
    """
    Suavização exponencial (Exponential Moving Average).

    Comparado à média móvel simples usada antes (`np.mean` de uma janela
    de N frames), o EMA reage mais rápido a mudanças reais de olhar
    (menos atraso perceptível) e ainda reduz o tremor frame a frame,
    porque dá peso maior às amostras recentes sem descartar o histórico.

    alpha próximo de 1.0 = quase sem suavização (responsivo, mais tremido)
    alpha próximo de 0.0 = muito suave (estável, mas com mais atraso)
    """

    def __init__(self, alpha: float = 0.35):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha deve estar entre 0 (exclusive) e 1 (inclusive)")
        self.alpha = alpha
        self.valor: float | None = None

    def atualizar(self, novo_valor: float) -> float:
        if self.valor is None:
            self.valor = novo_valor
        else:
            self.valor = self.alpha * novo_valor + (1 - self.alpha) * self.valor
        return self.valor

    def reset(self) -> None:
        self.valor = None


# =====================================================================
# Classificação esquerda/direita/centro
# =====================================================================
def classificar_lado(ratio_x: float, media_esq: float, media_dir: float) -> str:
    """Classifica por vizinho mais próximo entre os dois pontos calibrados (SIM e NAO)."""
    ponto_medio = (media_esq + media_dir) / 2
    metade_intervalo = abs(media_dir - media_esq) / 2
    zona_morta = metade_intervalo * 0.3
    esquerda_e_menor = media_esq < media_dir

    if ratio_x < ponto_medio - zona_morta:
        return "Esquerda" if esquerda_e_menor else "Direita"
    elif ratio_x > ponto_medio + zona_morta:
        return "Direita" if esquerda_e_menor else "Esquerda"
    return "Centro"


# =====================================================================
# EAR (Eye Aspect Ratio) -- detecção de piscada, usada como confirmação
# alternativa ao dwell time (olhar fixo por N frames)
# =====================================================================
def calcular_ear(landmarks, indices_verticais, indices_cantos, largura, altura) -> float:
    """
    Eye Aspect Ratio: cai quando o olho fecha (pisca). Valor tipicamente
    acima de ~0.25 com olho aberto e abaixo de ~0.15 com olho fechado,
    mas isso varia por pessoa/câmera -- o limiar é configurável
    (config/settings.yaml -> deteccao.piscada.limiar_ear).
    """
    p_sup = landmarks[indices_verticais[0]]
    p_inf = landmarks[indices_verticais[1]]
    p_canto1 = landmarks[indices_cantos[0]]
    p_canto2 = landmarks[indices_cantos[1]]

    dist_vertical = abs(p_sup.y - p_inf.y) * altura
    dist_horizontal = abs(p_canto1.x - p_canto2.x) * largura

    return dist_vertical / max(dist_horizontal, 1e-3)


def olho_fechado(ear: float, limiar: float = 0.15) -> bool:
    return ear < limiar


# =====================================================================
# Compensação de pose de cabeça
# =====================================================================
def extrair_yaw_da_matriz(matriz_transformacao: np.ndarray) -> float:
    """
    Extrai o ângulo de yaw (rotação esquerda/direita da cabeça) em graus,
    a partir da matriz de transformação facial 4x4 que o MediaPipe retorna
    quando `output_facial_transformation_matrixes=True`.

    Usado para não confundir uma cabeça levemente virada com um
    movimento real dos olhos.
    """
    r = matriz_transformacao[:3, :3]
    yaw_rad = np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    return float(np.degrees(yaw_rad))


def compensar_ratio_por_yaw(ratio_x: float, yaw_graus: float, sensibilidade: float = 0.004) -> float:
    """
    Ajusta o ratio_x horizontal subtraindo uma fração proporcional ao
    ângulo de yaw da cabeça. `sensibilidade` foi escolhida de forma
    conservadora (efeito pequeno por grau) -- calibre empiricamente
    observando o gráfico temporal ao girar a cabeça sem mover os olhos.
    """
    correcao = yaw_graus * sensibilidade
    return min(max(ratio_x - correcao, 0.0), 1.0)
