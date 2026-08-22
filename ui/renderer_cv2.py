"""
Camada de apresentação (OpenCV).

Tudo que desenha na tela mora aqui. Nada neste arquivo calcula gaze ou
decide transição de estado -- ele só recebe números/estado já prontos
e desenha. Isso permite, no futuro, trocar esse arquivo por uma UI Qt
ou por eventos WebSocket para um front-end web, sem tocar no restante
do sistema (ver core/gaze_engine.py e core/state_machine.py).
"""
from __future__ import annotations

import cv2
import numpy as np


def extrair_roi_do_olho(frame, landmarks, indices_cantos, indices_iris, largura, altura):
    """
    Monta um recorte (ROI) da região do olho para exibição de debug,
    com um marcador verde na posição estimada da íris.
    """
    p1 = landmarks[indices_cantos[0]]
    p2 = landmarks[indices_cantos[1]]
    x1_px, y1_px = p1.x * largura, p1.y * altura
    x2_px, y2_px = p2.x * largura, p2.y * altura

    xs_iris = [landmarks[i].x * largura for i in indices_iris]
    ys_iris = [landmarks[i].y * altura for i in indices_iris]
    iris_cx = float(np.mean(xs_iris))
    iris_cy = float(np.mean(ys_iris))

    x_esq, x_dir = min(x1_px, x2_px), max(x1_px, x2_px)
    largura_olho = max(x_dir - x_esq, 1e-3)
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
    return roi


def desenhar_recortes_debug(frame, roi_esq, roi_dir):
    if roi_esq.size > 0:
        frame[20:110, 20:160] = cv2.resize(roi_esq, (140, 90))
        cv2.rectangle(frame, (20, 20), (160, 110), (255, 255, 255), 1)
    if roi_dir.size > 0:
        frame[20:110, 170:310] = cv2.resize(roi_dir, (140, 90))
        cv2.rectangle(frame, (170, 20), (310, 110), (255, 255, 255), 1)


def desenhar_grafico_temporal(frame, hist_esq, hist_dir, media_esq, media_dir,
                               largura_grafico, altura_grafico, x, y, tamanho_historico):
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
            x1 = int(x + (i - 1) * largura_grafico / tamanho_historico)
            x2 = int(x + i * largura_grafico / tamanho_historico)
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


def desenhar_aviso_sem_rosto(frame, largura, altura):
    cv2.rectangle(frame, (0, altura // 2 - 40), (largura, altura // 2 + 40), (0, 0, 100), -1)
    cv2.putText(frame, "Rosto nao detectado -- reposicione-se em frente a camera",
                (largura // 2 - 420, altura // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)


def desenhar_menu(frame, largura, altura, calibracao_ja_feita, media_esq, media_dir):
    cv2.rectangle(frame, (0, 0), (largura, altura), (30, 30, 30), -1)
    cv2.putText(frame, "EYE-TRACKING ASSISTIVO", (largura // 2 - 260, altura // 2 - 150),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(frame, "[P]  Selecionar perfil", (largura // 2 - 150, altura // 2 - 60),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "[C]  Calibrar", (largura // 2 - 150, altura // 2 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "[I]  Iniciar SIM / NAO", (largura // 2 - 150, altura // 2 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "[Q]  Sair", (largura // 2 - 150, altura // 2 + 90),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    texto_status = (f"Ultima calibracao -> SIM: {media_esq:.2f}  NAO: {media_dir:.2f}"
                    if calibracao_ja_feita else "Calibracao ainda nao feita (usando valores padrao)")
    cv2.putText(frame, texto_status, (largura // 2 - 260, altura // 2 + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def desenhar_selecao_perfil(frame, largura, altura, perfis: list[str]):
    cv2.rectangle(frame, (0, 0), (largura, altura), (25, 25, 40), -1)
    cv2.putText(frame, "SELECIONAR PERFIL", (largura // 2 - 220, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    if not perfis:
        cv2.putText(frame, "Nenhum perfil salvo ainda.", (largura // 2 - 200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
    for i, nome in enumerate(perfis[:9]):
        cv2.putText(frame, f"[{i + 1}]  {nome}", (largura // 2 - 150, 150 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "[N] Novo perfil    [M] Voltar ao menu", (largura // 2 - 220, altura - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)


def desenhar_selecionado(frame, largura, altura, opcao_confirmada):

    # Texto que será exibido
    texto = f"SELECIONADO: {opcao_confirmada}"
    
    # Configurações de fonte e tamanho
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.7
    espessura = 2
    
    # Obtém o tamanho (largura e altura) que o texto ocupará na tela
    (largura_texto, altura_texto), linha_base = cv2.getTextSize(texto, fonte, escala, espessura)
    padding = 10
    
    # Define as coordenadas do retângulo no canto superior direito
    # Margem de 20 pixels a partir da borda direita e do topo
    x2 = largura - 20
    x1 = x2 - largura_texto - (padding * 2)
    y1 = 20
    y2 = y1 + altura_texto + (padding * 2)
    
    # Desenha o fundo da caixinha
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Preenchimento branco
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)         # Borda preta
    
    # Posiciona e desenha o texto centralizado dentro da caixinha
    cv2.putText(frame, texto, (x1 + padding, y2 - padding - 5), fonte, escala, (0, 0, 0), espessura)