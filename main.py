"""
Eye-Tracking Assistivo V2.1 -- ponto de entrada principal.

Este arquivo só ORQUESTRA os módulos abaixo; a lógica pesada vive em
cada um deles:

  config/loader.py        -> configuração externa (YAML)
  core/logging_config.py  -> logging estruturado
  core/model_loader.py     -> download/verificação do modelo MediaPipe
  core/camera_thread.py    -> captura de câmera em thread separada
  core/state_machine.py    -> estado da aplicação (Enum + AppState)
  core/gaze_engine.py      -> cálculo puro de gaze (ratio, EMA, EAR, pose)
  core/calibration.py      -> persistência de calibração por perfil
  ui/renderer_cv2.py       -> tudo que desenha na tela (OpenCV)

Rode com: python main.py
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config.loader import carregar_config
from core import gaze_engine as ge
from core.calibration import GerenciadorCalibracao
from core.camera_thread import CameraIndisponivelError, CameraStream
from core.logging_config import configurar_logging
from core.model_loader import garantir_modelo_baixado
from core.state_machine import AppState, EstadoApp
from ui import renderer_cv2 as ui


def montar_detector(caminho_modelo: str):
    base_options = mp_python.BaseOptions(model_asset_path=caminho_modelo)
    opcoes = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,  # necessário p/ compensação de pose
    )
    return mp_vision.FaceLandmarker.create_from_options(opcoes)


def main() -> None:
    cfg = carregar_config()
    logger = configurar_logging(pasta=cfg["logs"]["pasta"], nivel=cfg["logs"]["nivel"])

    det_cfg = cfg["deteccao"]
    cam_cfg = cfg["camera"]
    modelo_cfg = cfg["modelo"]
    janela_cfg = cfg["janela"]
    piscada_cfg = det_cfg["piscada"]

    caminho_modelo = garantir_modelo_baixado(
        modelo_cfg["caminho"], modelo_cfg["url"], modelo_cfg.get("sha256_esperado", "")
    )
    detector = montar_detector(str(caminho_modelo))

    try:
        camera = CameraStream(cam_cfg["indice"], cam_cfg["largura"], cam_cfg["altura"])
    except CameraIndisponivelError as e:
        logger.error(str(e))
        print(f"ERRO: {e}")
        return

    calibrador = GerenciadorCalibracao(pasta=cfg["perfis"]["pasta"])

    app = AppState()
    suavizador_esq = ge.SuavizadorEMA(alpha=det_cfg["alpha_ema"])
    suavizador_dir = ge.SuavizadorEMA(alpha=det_cfg["alpha_ema"])

    historico_esq = deque(maxlen=det_cfg["tamanho_historico_grafico"])
    historico_dir = deque(maxlen=det_cfg["tamanho_historico_grafico"])

    largura_grafico, altura_grafico = 280, 160

    cv2.namedWindow(janela_cfg["titulo"], cv2.WND_PROP_FULLSCREEN)
    if janela_cfg["fullscreen"]:
        cv2.setWindowProperty(janela_cfg["titulo"], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logger.info("Aplicação iniciada. Estado inicial: %s", app.estado)

    try:
        while True:
            try:
                frame = camera.ler(timeout=2.0)
            except Exception:
                logger.warning("Nenhum frame recebido da câmera em 2s.")
                continue

            frame = cv2.flip(frame, 1)
            altura, largura, _ = frame.shape
            metade_x = largura // 2
            pos_x = largura - largura_grafico - 20
            pos_y = altura - altura_grafico - 20
            overlay = frame.copy()
            cor_sim, cor_nao = (0, 60, 0), (0, 0, 60)

            ratio_esq, ratio_dir = 0.5, 0.5
            lado_detectado = "Centro"
            lado_esq, lado_dir = "Centro", "Centro"
            rosto_detectado = False

            rodar_deteccao = app.estado in (EstadoApp.CALIB_COLETANDO, EstadoApp.ATIVO)

            if rodar_deteccao:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.time() * 1000)
                resultado = detector.detect_for_video(mp_image, timestamp_ms)

                if resultado.face_landmarks:
                    rosto_detectado = True
                    app.contador_sem_rosto = 0
                    landmarks = resultado.face_landmarks[0]

                    ratio_esq_bruto = ge.calcular_ratio_iris_x(
                        landmarks, ge.OLHO_ESQUERDO_CANTOS, ge.IRIS_ESQUERDA, largura, altura)
                    ratio_dir_bruto = ge.calcular_ratio_iris_x(
                        landmarks, ge.OLHO_DIREITO_CANTOS, ge.IRIS_DIREITA, largura, altura)

                    # Compensação de pose de cabeça, se a matriz de transformação vier disponível
                    if resultado.facial_transformation_matrixes:
                        yaw = ge.extrair_yaw_da_matriz(resultado.facial_transformation_matrixes[0])
                        ratio_esq_bruto = ge.compensar_ratio_por_yaw(ratio_esq_bruto, yaw)
                        ratio_dir_bruto = ge.compensar_ratio_por_yaw(ratio_dir_bruto, yaw)

                    ratio_esq = suavizador_esq.atualizar(ratio_esq_bruto)
                    ratio_dir = suavizador_dir.atualizar(ratio_dir_bruto)

                    lado_esq = ge.classificar_lado(ratio_esq, app.media_esquerda_calibrada, app.media_direita_calibrada)
                    lado_dir = ge.classificar_lado(ratio_dir, app.media_esquerda_calibrada, app.media_direita_calibrada)

                    ratio_valido = (ratio_esq + ratio_dir) / 2
                    historico_esq.append(ratio_esq)
                    historico_dir.append(ratio_dir)

                    if app.estado == EstadoApp.ATIVO:
                        lados = [lado_esq, lado_dir]
                        if "Esquerda" in lados and "Direita" not in lados:
                            lado_detectado = "Esquerda"
                        elif "Direita" in lados and "Esquerda" not in lados:
                            lado_detectado = "Direita"

                        # Confirmação alternativa por piscada
                        if piscada_cfg["habilitada"]:
                            ear_esq = ge.calcular_ear(landmarks, ge.OLHO_ESQUERDO_VERTICAL, ge.OLHO_ESQUERDO_CANTOS, largura, altura)
                            ear_dir = ge.calcular_ear(landmarks, ge.OLHO_DIREITO_VERTICAL, ge.OLHO_DIREITO_CANTOS, largura, altura)
                            ear_medio = (ear_esq + ear_dir) / 2
                            if ge.olho_fechado(ear_medio, piscada_cfg["limiar_ear"]):
                                app.contador_piscada += 1
                            else:
                                app.contador_piscada = 0

                    if app.estado == EstadoApp.CALIB_COLETANDO:
                        app.coletas_calibracao.append(ratio_valido)

                    roi_esq = ui.extrair_roi_do_olho(frame, landmarks, ge.OLHO_ESQUERDO_CANTOS, ge.IRIS_ESQUERDA, largura, altura)
                    roi_dir = ui.extrair_roi_do_olho(frame, landmarks, ge.OLHO_DIREITO_CANTOS, ge.IRIS_DIREITA, largura, altura)
                    ui.desenhar_recortes_debug(frame, roi_esq, roi_dir)

                if not rosto_detectado:
                    app.contador_sem_rosto += 1
                    if app.contador_sem_rosto > det_cfg["limite_frames_sem_rosto"]:
                        ui.desenhar_aviso_sem_rosto(frame, largura, altura)

            tecla = cv2.waitKey(1) & 0xFF

            # =============================================================
            if app.estado == EstadoApp.MENU:
                ui.desenhar_menu(frame, largura, altura, app.calibracao_ja_feita,
                                  app.media_esquerda_calibrada, app.media_direita_calibrada)
                if tecla == ord('p'):
                    app.transicionar(EstadoApp.SELECIONAR_PERFIL)
                elif tecla == ord('c'):
                    app.transicionar(EstadoApp.CALIB_PREPARAR)
                    app.resetar_calibracao()
                    suavizador_esq.reset()
                    suavizador_dir.reset()
                elif tecla == ord('i'):
                    app.transicionar(EstadoApp.ATIVO)
                    app.resetar_selecao()
                    suavizador_esq.reset()
                    suavizador_dir.reset()

            # =============================================================
            elif app.estado == EstadoApp.SELECIONAR_PERFIL:
                perfis = calibrador.listar_perfis()
                ui.desenhar_selecao_perfil(frame, largura, altura, perfis)
                if tecla == ord('m'):
                    app.transicionar(EstadoApp.MENU)
                elif tecla == ord('n'):
                    logger.info("Novo perfil solicitado (defina o nome via terminal/config).")
                    app.transicionar(EstadoApp.MENU)
                elif ord('1') <= tecla <= ord('9'):
                    idx = tecla - ord('1')
                    if idx < len(perfis):
                        nome = perfis[idx]
                        dados = calibrador.carregar(nome)
                        if dados:
                            app.perfil_ativo = nome
                            app.media_esquerda_calibrada = dados["media_esquerda"]
                            app.media_direita_calibrada = dados["media_direita"]
                            app.calibracao_ja_feita = True
                            if calibrador.calibracao_esta_vencida(dados, cfg["perfis"]["validade_dias"]):
                                logger.info("Calibração de '%s' está vencida -- considere recalibrar.", nome)
                            logger.info("Perfil '%s' carregado.", nome)
                            app.transicionar(EstadoApp.MENU)

            # =============================================================
            elif app.estado == EstadoApp.CALIB_PREPARAR:
                alvo = "SIM" if app.calib_fase == "ESQUERDA" else "NAO"
                cor_alvo = (0, 255, 0) if app.calib_fase == "ESQUERDA" else (0, 0, 255)
                cor_sim_c = (0, 150, 0) if app.calib_fase == "ESQUERDA" else (0, 60, 0)
                cor_nao_c = (0, 0, 150) if app.calib_fase == "DIREITA" else (0, 0, 60)
                ui.desenhar_painel_sim_nao(frame, overlay, largura, altura, metade_x, cor_sim_c, cor_nao_c, 0.35)
                cv2.putText(frame, "FASE DE CALIBRACAO", (metade_x - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Posicione o olhar fixo para o '{alvo}'", (largura // 2 - 240, 110),
                            cv2.FONT_HERSHEY_DUPLEX, 1, cor_alvo, 2)
                cv2.putText(frame, "Aperte ESPACO para comecar a coleta", (largura // 2 - 240, altura - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, "[M] Voltar ao menu", (largura // 2 - 60, altura - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

                if tecla == 32:
                    app.transicionar(EstadoApp.CALIB_COLETANDO)
                    app.coletas_calibracao = []
                elif tecla == ord('m'):
                    app.transicionar(EstadoApp.MENU)

            # =============================================================
            elif app.estado == EstadoApp.CALIB_COLETANDO:
                alvo = "SIM" if app.calib_fase == "ESQUERDA" else "NAO"
                cor_sim_c = (0, 200, 0) if app.calib_fase == "ESQUERDA" else (0, 60, 0)
                cor_nao_c = (0, 0, 200) if app.calib_fase == "DIREITA" else (0, 0, 60)
                ui.desenhar_painel_sim_nao(frame, overlay, largura, altura, metade_x, cor_sim_c, cor_nao_c, 0.35)
                cv2.putText(frame, "FASE DE CALIBRACAO", (largura // 2 - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"OLHE FIXO PARA O '{alvo}'", (50, 95), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

                progresso = int((len(app.coletas_calibracao) / det_cfg["num_frames_calibracao"]) * (metade_x - 100))
                x_barra = 50 if app.calib_fase == "ESQUERDA" else metade_x + 50
                cv2.rectangle(frame, (x_barra, 130), (x_barra + progresso, 145), (0, 255, 0), -1)

                ui.desenhar_grafico_temporal(frame, historico_esq, historico_dir,
                                             app.media_esquerda_calibrada, app.media_direita_calibrada,
                                             largura_grafico, altura_grafico, pos_x, pos_y,
                                             det_cfg["tamanho_historico_grafico"])

                if len(app.coletas_calibracao) >= det_cfg["num_frames_calibracao"]:
                    if app.calib_fase == "ESQUERDA":
                        app.media_esquerda_calibrada = float(np.mean(app.coletas_calibracao))
                        app.calib_fase = "DIREITA"
                        app.coletas_calibracao = []
                        app.transicionar(EstadoApp.CALIB_PREPARAR)
                    else:
                        app.media_direita_calibrada = float(np.mean(app.coletas_calibracao))
                        app.coletas_calibracao = []
                        app.calibracao_ja_feita = True

                        diferenca = abs(app.media_direita_calibrada - app.media_esquerda_calibrada)
                        if diferenca < det_cfg["diferenca_minima_calibracao"]:
                            logger.warning(
                                "Diferença entre SIM (%.3f) e NAO (%.3f) foi de apenas %.3f. "
                                "Recomenda-se recalibrar pedindo um olhar mais amplo.",
                                app.media_esquerda_calibrada, app.media_direita_calibrada, diferenca,
                            )
                        logger.info("Calibrado! SIM: %.3f | NAO: %.3f",
                                    app.media_esquerda_calibrada, app.media_direita_calibrada)

                        nome_perfil = app.perfil_ativo or "padrao"
                        calibrador.salvar(nome_perfil, app.media_esquerda_calibrada, app.media_direita_calibrada)
                        app.transicionar(EstadoApp.CALIB_CONCLUIDA)

            # =============================================================
            elif app.estado == EstadoApp.CALIB_CONCLUIDA:
                cv2.rectangle(frame, (0, 0), (largura, altura), (20, 60, 20), -1)
                cv2.putText(frame, "CALIBRACAO CONCLUIDA!", (largura // 2 - 260, altura // 2 - 60),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, f"SIM: {app.media_esquerda_calibrada:.2f}   NAO: {app.media_direita_calibrada:.2f}",
                            (largura // 2 - 220, altura // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "Aperte ENTER para confirmar e voltar ao menu",
                            (largura // 2 - 320, altura // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                if tecla in (13, 10):
                    app.transicionar(EstadoApp.MENU)

            # =============================================================
            elif app.estado == EstadoApp.ATIVO:
                if lado_detectado == "Esquerda":
                    cor_sim = (0, 180, 0)
                elif lado_detectado == "Direita":
                    cor_nao = (0, 0, 220)
                ui.desenhar_painel_sim_nao(frame, overlay, largura, altura, metade_x, cor_sim, cor_nao, 0.25)
                cv2.putText(frame, "[M] Voltar ao menu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                texto_debug = (f"esq:{ratio_esq:.2f} dir:{ratio_dir:.2f} | "
                                f"calib SIM:{app.media_esquerda_calibrada:.2f} NAO:{app.media_direita_calibrada:.2f}")
                cv2.putText(frame, texto_debug, (10, altura - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                ui.desenhar_grafico_temporal(frame, historico_esq, historico_dir,
                                             app.media_esquerda_calibrada, app.media_direita_calibrada,
                                             largura_grafico, altura_grafico, pos_x, pos_y,
                                             det_cfg["tamanho_historico_grafico"])

                if not app.opcao_confirmada:
                    opcao_atual = {"Esquerda": "SIM", "Direita": "NAO"}.get(lado_detectado)

                    if opcao_atual and opcao_atual == app.opcao_focada:
                        app.contador_fixacao += 1
                    else:
                        app.contador_fixacao = 0
                        app.opcao_focada = opcao_atual

                    confirmou_por_dwell = app.contador_fixacao >= det_cfg["frames_para_selecionar"]
                    confirmou_por_piscada = (
                        piscada_cfg["habilitada"]
                        and app.opcao_focada is not None
                        and app.contador_piscada >= piscada_cfg["frames_minimos_piscada"]
                    )

                    if confirmou_por_dwell or confirmou_por_piscada:
                        app.opcao_confirmada = app.opcao_focada
                        app.tempo_selecionado = time.time()
                        app.contador_fixacao = 0
                        app.contador_piscada = 0
                        logger.info("Seleção confirmada: %s (método=%s)",
                                    app.opcao_confirmada, "piscada" if confirmou_por_piscada else "dwell")

                    if app.opcao_focada:
                        progresso = int((app.contador_fixacao / det_cfg["frames_para_selecionar"]) * metade_x)
                        y_barra = altura - 30
                        if app.opcao_focada == "SIM":
                            cv2.rectangle(frame, (0, y_barra), (progresso, y_barra + 15), (0, 255, 0), -1)
                        else:
                            cv2.rectangle(frame, (largura - progresso, y_barra), (largura, y_barra + 15), (0, 0, 255), -1)
                else:
                    if time.time() - app.tempo_selecionado < 1.5:
                        ui.desenhar_selecionado(frame, largura, altura, app.opcao_confirmada)
                    else:
                        app.resetar_selecao()

                if tecla == ord('m'):
                    app.resetar_selecao()
                    app.transicionar(EstadoApp.MENU)

            cv2.imshow(janela_cfg["titulo"], frame)

            if tecla == ord('q'):
                logger.info("Encerrando por solicitação do usuário.")
                break

    finally:
        camera.parar()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
