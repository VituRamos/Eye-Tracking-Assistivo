import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =====================================================================
# 1. RASTREAMENTO DE OLHAR COM MEDIAPIPE FACE LANDMARKER (TASKS API)
# =====================================================================
# A partir de certas versões do mediapipe (ex: 0.10.35 em Python 3.12+),
# a API clássica "mp.solutions.face_mesh" foi descontinuada e substituída
# pela Tasks API ("mp.tasks.vision.FaceLandmarker"). Os ÍNDICES dos
# landmarks (olhos/íris) são os mesmos de antes -- só muda a forma de
# inicializar o detector e de rodar a detecção a cada frame.

CAMINHO_MODELO = "face_landmarker.task"
URL_MODELO = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def garantir_modelo_baixado(caminho=CAMINHO_MODELO, url=URL_MODELO):
    """Baixa o modelo .task do Face Landmarker na primeira execução, se ainda não existir localmente."""
    if not os.path.exists(caminho):
        print(f"Baixando modelo do MediaPipe Face Landmarker para '{caminho}' ...")
        urllib.request.urlretrieve(url, caminho)
        print("Download concluído.")

garantir_modelo_baixado()

base_options = mp_python.BaseOptions(model_asset_path=CAMINHO_MODELO)
opcoes_landmarker = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
detector_landmarks = mp_vision.FaceLandmarker.create_from_options(opcoes_landmarker)

# Índices oficiais do MediaPipe Face Mesh (canônicos, não mudam entre rostos)
OLHO_DIREITO_CANTOS = (33, 133)     # (canto externo, canto interno) do olho direito
OLHO_ESQUERDO_CANTOS = (362, 263)   # (canto interno, canto externo) do olho esquerdo
IRIS_DIREITA = [469, 470, 471, 472]
IRIS_ESQUERDA = [474, 475, 476, 477]

# --- CONFIGURAÇÕES DO SISTEMA ---
FRAMES_PARA_SELECIONAR = 15   # Quantos frames olhando a opção para confirmar (~0.7s)
TAMANHO_FILTRO = 5            # Frames usados para suavizar o tremor do sinal (ratio_x)
largura_grafico = 160
altura_grafico = 60

# --- CONFIGURAÇÕES DE CALIBRAÇÃO DINÂMICA ---
NUM_FRAMES_CALIBRACAO = 90
DIFERENCA_MINIMA_CALIBRACAO = 0.03

media_esquerda_calibrada = 0.35
media_direita_calibrada = 0.65
calibracao_ja_feita = False

# Suavização do sinal (ratio_x de cada olho) - reduz tremor frame a frame
hist_ratio_esq, hist_ratio_dir = [], []

# Histórico do sinal para o gráfico temporal ao vivo
TAMANHO_HISTORICO_GRAFICO = 150
historico_ratio_esq = deque(maxlen=TAMANHO_HISTORICO_GRAFICO)
historico_ratio_dir = deque(maxlen=TAMANHO_HISTORICO_GRAFICO)

# --- MÁQUINA DE ESTADOS DO APLICATIVO ---
estado_app = "MENU"
calib_fase = "ESQUERDA"
coletas_calibracao = []

contador_fixacao = 0
opcao_focada = None
opcao_confirmada = None
tempo_selecionado = 0

def suavizar_valor(historico, novo_valor, max_itens=TAMANHO_FILTRO):
    """ Média móvel simples para reduzir o tremor do sinal ratio_x """
    historico.append(novo_valor)
    if len(historico) > max_itens:
        historico.pop(0)
    return float(np.mean(historico))

def limpar_historicos():
    hist_ratio_esq.clear()
    hist_ratio_dir.clear()

def calcular_ratio_e_recorte(frame, landmarks, indices_cantos, indices_iris, largura, altura):
    """
    Calcula a posição horizontal da íris DENTRO da largura real daquele olho
    (0.0 = encostada no canto A, 1.0 = encostada no canto B), e monta um
    recorte da região para exibição de debug.
    """
    p1 = landmarks[indices_cantos[0]]
    p2 = landmarks[indices_cantos[1]]
    x1_px, y1_px = p1.x * largura, p1.y * altura
    x2_px, y2_px = p2.x * largura, p2.y * altura

    xs_iris = [landmarks[i].x * largura for i in indices_iris]
    ys_iris = [landmarks[i].y * altura for i in indices_iris]
    iris_cx = float(np.mean(xs_iris))
    iris_cy = float(np.mean(ys_iris))

    x_esq = min(x1_px, x2_px)
    x_dir = max(x1_px, x2_px)
    largura_olho = max(x_dir - x_esq, 1e-3)

    ratio_x = (iris_cx - x_esq) / largura_olho
    ratio_x = min(max(ratio_x, 0.0), 1.0)

    cy_medio = (y1_px + y2_px) / 2
    margem_y = largura_olho * 0.7
    margem_x = largura_olho * 0.2
    x1c = int(max(0, x_esq - margem_x))
    x2c = int(min(largura, x_dir + margem_x))
    y1c = int(max(0, cy_medio - margem_y))
    y2c = int(min(altura, cy_medio + margem_y))

    roi = frame[y1c:y2c, x1c:x2c].copy()
    if roi.size > 0:
        px = int(min(max(iris_cx - x1c, 0), roi.shape[1] - 1))
        py = int(min(max(iris_cy - y1c, 0), roi.shape[0] - 1))
        cv2.circle(roi, (px, py), 2, (0, 255, 0), -1)

    return ratio_x, roi

def classificar_lado(ratio_x, media_esq, media_dir):
    """ Classifica por vizinho mais próximo entre os dois pontos calibrados (SIM e NAO) """
    ponto_medio = (media_esq + media_dir) / 2
    metade_intervalo = abs(media_dir - media_esq) / 2
    zona_morta = metade_intervalo * 0.3
    esquerda_e_menor = media_esq < media_dir

    if ratio_x < ponto_medio - zona_morta:
        return "Esquerda" if esquerda_e_menor else "Direita"
    elif ratio_x > ponto_medio + zona_morta:
        return "Direita" if esquerda_e_menor else "Esquerda"
    return "Centro"

def desenhar_grafico_temporal(frame, hist_esq, hist_dir, media_esq, media_dir, largura_grafico, altura_grafico, x, y):
    cv2.rectangle(frame, (x, y), (x + largura_grafico, y + altura_grafico), (15, 15, 15), -1)
    cv2.rectangle(frame, (x, y), (x + largura_grafico, y + altura_grafico), (255, 255, 255), 1)

    def valor_para_y(v):
        v = min(max(v, 0.0), 1.0)
        return int(y + altura_grafico - (v * altura_grafico))

    ponto_medio = (media_esq + media_dir) / 2
    cv2.line(frame, (x, valor_para_y(media_esq)), (x + largura_grafico, valor_para_y(media_esq)), (0, 200, 0), 1)
    cv2.line(frame, (x, valor_para_y(media_dir)), (x + largura_grafico, valor_para_y(media_dir)), (0, 0, 220), 1)
    for xi in range(x, x + largura_grafico, 8):
        cv2.line(frame, (xi, valor_para_y(ponto_medio)), (min(xi + 4, x + largura_grafico), valor_para_y(ponto_medio)), (130, 130, 130), 1)

    def desenhar_serie(historico, cor):
        pontos = list(historico)
        n = len(pontos)
        if n < 2:
            return
        for i in range(1, n):
            x1 = int(x + (i - 1) * largura_grafico / TAMANHO_HISTORICO_GRAFICO)
            x2 = int(x + i * largura_grafico / TAMANHO_HISTORICO_GRAFICO)
            y1 = valor_para_y(pontos[i - 1])
            y2 = valor_para_y(pontos[i])
            cv2.line(frame, (x1, y1), (x2, y2), cor, 1)

    desenhar_serie(hist_esq, (255, 255, 0))
    desenhar_serie(hist_dir, (255, 0, 255))

    cv2.putText(frame, "Sinal temporal (ratio_x da iris)", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(frame, "olho esq", (x + 8, y + altura_grafico - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 0), 1)
    cv2.putText(frame, "olho dir", (x + 90, y + altura_grafico - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 0, 255), 1)
    cv2.putText(frame, "ref. SIM", (x + largura_grafico - 150, y + altura_grafico - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 200, 0), 1)
    cv2.putText(frame, "ref. NAO", (x + largura_grafico - 75, y + altura_grafico - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 220), 1)

def desenhar_painel_sim_nao(frame, overlay, largura, altura, metade_x, cor_sim, cor_nao, intensidade=0.25):
    cv2.rectangle(overlay, (0, 0), (metade_x, altura), cor_sim, -1)
    cv2.rectangle(overlay, (metade_x, 0), (largura, altura), cor_nao, -1)
    cv2.addWeighted(overlay, intensidade, frame, 1 - intensidade, 0, frame)
    cv2.line(frame, (metade_x, 0), (metade_x, altura), (255, 255, 255), 2)
    cv2.putText(frame, "SIM", (metade_x // 2 - 50, altura // 2), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "NAO", (metade_x + (metade_x // 2) - 50, altura // 2), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)

# =====================================================================
# 2. LOOP PRINCIPAL
# =====================================================================
cap = cv2.VideoCapture(0)

cv2.namedWindow("Eye-Tracking Assistivo V2.0 (MediaPipe)", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Eye-Tracking Assistivo V2.0 (MediaPipe)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    sucesso, frame = cap.read()
    if not sucesso:
        break

    frame = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape
    metade_x = largura // 2
    pos_x = largura - largura_grafico - 20
    pos_y = altura - altura_grafico - 20
    overlay = frame.copy()
    cor_sim = (0, 60, 0)
    cor_nao = (0, 0, 60)

    ratio_esq, ratio_dir = 0.5, 0.5
    lado_detectado = "Centro"
    ratio_valido = None
    roi_esq, roi_dir = np.array([]), np.array([])
    lado_esq, lado_dir = "Centro", "Centro"

    rodar_deteccao = estado_app in ("CALIB_COLETANDO", "ATIVO")

    if rodar_deteccao:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        resultado = detector_landmarks.detect_for_video(mp_image, timestamp_ms)

        if resultado.face_landmarks:
            landmarks = resultado.face_landmarks[0]

            ratio_esq_bruto, roi_esq = calcular_ratio_e_recorte(
                frame, landmarks, OLHO_ESQUERDO_CANTOS, IRIS_ESQUERDA, largura, altura)
            ratio_dir_bruto, roi_dir = calcular_ratio_e_recorte(
                frame, landmarks, OLHO_DIREITO_CANTOS, IRIS_DIREITA, largura, altura)

            ratio_esq = suavizar_valor(hist_ratio_esq, ratio_esq_bruto)
            ratio_dir = suavizar_valor(hist_ratio_dir, ratio_dir_bruto)

            lado_esq = classificar_lado(ratio_esq, media_esquerda_calibrada, media_direita_calibrada)
            lado_dir = classificar_lado(ratio_dir, media_esquerda_calibrada, media_direita_calibrada)

            ratio_valido = (ratio_esq + ratio_dir) / 2

            historico_ratio_esq.append(ratio_esq)
            historico_ratio_dir.append(ratio_dir)

            if estado_app == "ATIVO":
                lados_atuais = [lado_esq, lado_dir]
                if "Esquerda" in lados_atuais and "Direita" not in lados_atuais:
                    lado_detectado = "Esquerda"
                elif "Direita" in lados_atuais and "Esquerda" not in lados_atuais:
                    lado_detectado = "Direita"

            if estado_app == "CALIB_COLETANDO" and ratio_valido is not None:
                coletas_calibracao.append(ratio_valido)

            if roi_esq.size > 0:
                roi_esq_redim = cv2.resize(roi_esq, (140, 90))
                frame[20:110, 20:160] = roi_esq_redim
                cv2.rectangle(frame, (20, 20), (160, 110), (255, 255, 255), 1)
            if roi_dir.size > 0:
                roi_dir_redim = cv2.resize(roi_dir, (140, 90))
                frame[20:110, 170:310] = roi_dir_redim
                cv2.rectangle(frame, (170, 20), (310, 110), (255, 255, 255), 1)

    tecla = cv2.waitKey(1) & 0xFF

    # =================================================================
    if estado_app == "MENU":
        cv2.rectangle(frame, (0, 0), (largura, altura), (30, 30, 30), -1)
        cv2.putText(frame, "EYE-TRACKING ASSISTIVO", (largura // 2 - 260, altura // 2 - 120),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        cv2.putText(frame, "[C]  Calibrar", (largura // 2 - 150, altura // 2 - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "[I]  Iniciar SIM / NAO", (largura // 2 - 150, altura // 2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "[Q]  Sair", (largura // 2 - 150, altura // 2 + 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        texto_status = (f"Ultima calibracao -> SIM: {media_esquerda_calibrada:.2f}  NAO: {media_direita_calibrada:.2f}"
                        if calibracao_ja_feita else "Calibracao ainda nao feita (usando valores padrao)")
        cv2.putText(frame, texto_status, (largura // 2 - 260, altura // 2 + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if tecla == ord('c'):
            estado_app = "CALIB_PREPARAR"
            calib_fase = "ESQUERDA"
            coletas_calibracao = []
            limpar_historicos()
        elif tecla == ord('i'):
            estado_app = "ATIVO"
            opcao_focada = None
            opcao_confirmada = None
            contador_fixacao = 0
            limpar_historicos()

    # =================================================================
    elif estado_app == "CALIB_PREPARAR":
        alvo = "SIM" if calib_fase == "ESQUERDA" else "NAO"
        cor_alvo = (0, 255, 0) if calib_fase == "ESQUERDA" else (0, 0, 255)
        cor_sim_calib = (0, 150, 0) if calib_fase == "ESQUERDA" else (0, 60, 0)
        cor_nao_calib = (0, 0, 150) if calib_fase == "DIREITA" else (0, 0, 60)
        desenhar_painel_sim_nao(frame, overlay, largura, altura, metade_x, cor_sim_calib, cor_nao_calib, 0.35)

        cv2.putText(frame, "FASE DE CALIBRACAO", (metade_x-160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Posicione o olhar fixo para o '{alvo}'", (largura // 2 - 240, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 1, cor_alvo, 2)
        cv2.putText(frame, "Quando estiver pronto, aperte ESPACO para comecar a coleta",
                    (largura // 2 - 280, altura - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, "[M] Voltar ao menu", (largura // 2 - 60, altura - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        if tecla == 32:
            estado_app = "CALIB_COLETANDO"
            coletas_calibracao = []
        elif tecla == ord('m'):
            estado_app = "MENU"

    # =================================================================
    elif estado_app == "CALIB_COLETANDO":
        alvo = "SIM" if calib_fase == "ESQUERDA" else "NAO"
        cor_sim_calib = (0, 200, 0) if calib_fase == "ESQUERDA" else (0, 60, 0)
        cor_nao_calib = (0, 0, 200) if calib_fase == "DIREITA" else (0, 0, 60)
        desenhar_painel_sim_nao(frame, overlay, largura, altura, metade_x, cor_sim_calib, cor_nao_calib, 0.35)

        cv2.putText(frame, "FASE DE CALIBRACAO", (largura // 2 - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"OLHE FIXO PARA O '{alvo}'", (50, 95), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        progresso = int((len(coletas_calibracao) / NUM_FRAMES_CALIBRACAO) * (metade_x - 100))
        x_barra = 50 if calib_fase == "ESQUERDA" else metade_x + 50
        cv2.rectangle(frame, (x_barra, 130), (x_barra + progresso, 145), (0, 255, 0), -1)

        desenhar_grafico_temporal(frame, historico_ratio_esq, historico_ratio_dir,
                                  media_esquerda_calibrada, media_direita_calibrada,
                                  largura_grafico, altura_grafico, pos_x, pos_y)

        if len(coletas_calibracao) >= NUM_FRAMES_CALIBRACAO:
            if calib_fase == "ESQUERDA":
                media_esquerda_calibrada = float(np.mean(coletas_calibracao))
                calib_fase = "DIREITA"
                coletas_calibracao = []
                estado_app = "CALIB_PREPARAR"
            else:
                media_direita_calibrada = float(np.mean(coletas_calibracao))
                coletas_calibracao = []
                calibracao_ja_feita = True

                diferenca = abs(media_direita_calibrada - media_esquerda_calibrada)
                if diferenca < DIFERENCA_MINIMA_CALIBRACAO:
                    print(f"-> AVISO: diferenca entre SIM ({media_esquerda_calibrada:.3f}) e "
                          f"NAO ({media_direita_calibrada:.3f}) foi de apenas {diferenca:.3f}. "
                          f"Recomenda-se recalibrar pedindo um olhar mais amplo.")
                print(f"-> Calibrado! SIM: {media_esquerda_calibrada:.3f} | NAO: {media_direita_calibrada:.3f}")
                estado_app = "CALIB_CONCLUIDA"

    # =================================================================
    elif estado_app == "CALIB_CONCLUIDA":
        cv2.rectangle(frame, (0, 0), (largura, altura), (20, 60, 20), -1)
        cv2.putText(frame, "CALIBRACAO CONCLUIDA!", (largura // 2 - 260, altura // 2 - 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"SIM: {media_esquerda_calibrada:.2f}   NAO: {media_direita_calibrada:.2f}",
                    (largura // 2 - 220, altura // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Aperte ENTER para confirmar e voltar ao menu",
                    (largura // 2 - 320, altura // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        if tecla in (13, 10):
            estado_app = "MENU"

    # =================================================================
    elif estado_app == "ATIVO":
        if lado_detectado == "Esquerda":
            cor_sim = (0, 180, 0)
        elif lado_detectado == "Direita":
            cor_nao = (0, 0, 220)
        desenhar_painel_sim_nao(frame, overlay, largura, altura, metade_x, cor_sim, cor_nao, 0.25)
        cv2.putText(frame, "[M] Voltar ao menu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        texto_debug = f"esq:{ratio_esq:.2f} dir:{ratio_dir:.2f} | calib SIM:{media_esquerda_calibrada:.2f} NAO:{media_direita_calibrada:.2f}"
        cv2.putText(frame, texto_debug, (10, altura - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        desenhar_grafico_temporal(frame, historico_ratio_esq, historico_ratio_dir,
                                  media_esquerda_calibrada, media_direita_calibrada,
                                  largura_grafico, altura_grafico, pos_x, pos_y)

        if not opcao_confirmada:
            if lado_detectado == "Esquerda":
                opcao_atual = "SIM"
            elif lado_detectado == "Direita":
                opcao_atual = "NAO"
            else:
                opcao_atual = None

            if opcao_atual and opcao_atual == opcao_focada:
                contador_fixacao += 1
            else:
                contador_fixacao = 0
                opcao_focada = opcao_atual

            if contador_fixacao >= FRAMES_PARA_SELECIONAR:
                opcao_confirmada = opcao_focada
                tempo_selecionado = time.time()
                contador_fixacao = 0

            if opcao_focada:
                progresso = int((contador_fixacao / FRAMES_PARA_SELECIONAR) * metade_x)
                y_barra = altura - 30
                if opcao_focada == "SIM":
                    cv2.rectangle(frame, (0, y_barra), (progresso, y_barra + 15), (0, 255, 0), -1)
                else:
                    cv2.rectangle(frame, (largura - progresso, y_barra), (largura, y_barra + 15), (0, 0, 255), -1)
        else:
            if time.time() - tempo_selecionado < 1.5:
                cv2.rectangle(frame, (0, 0), (largura, altura), (255, 255, 255), -1)
                cv2.putText(frame, f"SELECIONADO: {opcao_confirmada}", (largura // 6, altura // 2),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 3)
            else:
                opcao_confirmada = None
                opcao_focada = None
                estado_app = "ATIVO"

        if tecla == ord('m'):
            opcao_confirmada = None
            opcao_focada = None
            estado_app = "MENU"

    cv2.imshow("Eye-Tracking Assistivo V2.0 (MediaPipe)", frame)

    if tecla == ord('q'):
        break

cap.release()
detector_landmarks.close()
cv2.destroyAllWindows()
