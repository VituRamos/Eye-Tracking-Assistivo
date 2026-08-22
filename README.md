# Eye-Tracking Assistivo V2.1

Sistema de comunicação assistiva por rastreamento ocular (SIM / NÃO),
usando MediaPipe Face Landmarker. Esta versão reestrutura o protótipo
original em módulos testáveis e adiciona melhorias de engenharia e de
precisão descritas abaixo.

## Estrutura do projeto

```
eyetracking_assistivo/
├── config/
│   ├── settings.yaml       # todos os parâmetros ajustáveis (sem editar código)
│   └── loader.py           # leitura do YAML com fallback seguro
├── core/
│   ├── state_machine.py    # Enum de estados + AppState (substitui globais soltas)
│   ├── gaze_engine.py       # cálculo puro: ratio, EMA, EAR (piscada), pose de cabeça
│   ├── calibration.py       # persistência de calibração por perfil (JSON)
│   ├── camera_thread.py     # captura de câmera em thread separada
│   ├── model_loader.py       # download + verificação de integridade do modelo
│   └── logging_config.py    # logging estruturado (arquivo + console)
├── ui/
│   └── renderer_cv2.py       # tudo que desenha na tela (OpenCV) -- sem lógica de gaze
├── tests/
│   ├── test_gaze_engine.py   # testes das funções puras de cálculo
│   └── test_calibration.py   # testes de persistência de calibração
├── perfis/                   # calibrações salvas por usuário (gerado em runtime)
├── logs/                     # logs de sessão (gerado em runtime)
├── main.py                    # ponto de entrada -- só orquestra os módulos acima
├── requirements.txt
└── pytest.ini
```

## Instalação

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Executando

```bash
python main.py
```

Na primeira execução, o modelo `face_landmarker.task` é baixado
automaticamente (ver `core/model_loader.py`).

## Rodando os testes

```bash
pytest
```

Os testes em `tests/` cobrem apenas as partes **puras** do sistema
(cálculo de ratio, suavização EMA, classificação de lado, EAR/piscada,
compensação de pose e persistência de calibração) -- não dependem de
câmera nem do MediaPipe rodando de verdade, por isso rodam em qualquer
ambiente, inclusive CI.

## Controles

| Tecla | Ação |
|---|---|
| `P` | Selecionar perfil salvo |
| `C` | Iniciar calibração |
| `I` | Iniciar modo ativo (SIM/NÃO) |
| `M` | Voltar ao menu |
| `ESPAÇO` | Iniciar coleta durante a calibração |
| `ENTER` | Confirmar calibração concluída |
| `Q` | Sair |

## O que mudou em relação ao protótipo original

**Engenharia:**
- Código dividido em módulos com responsabilidade única (config / core / ui)
- Estado da aplicação centralizado em `AppState` (dataclass) em vez de
  ~10 variáveis globais soltas
- Configuração externa via `settings.yaml` -- ajustar sensibilidade,
  resolução de câmera, etc. sem tocar em Python
- Logging estruturado em arquivo por sessão, substituindo `print()`
- Calibração persistida por perfil de usuário em JSON, com data e
  indicador de qualidade
- Captura de câmera em thread separada (fila de tamanho 1), para a UI
  não travar se o processamento atrasar um frame
- Verificação opcional de checksum (SHA256) do modelo baixado
- Testes automatizados (`pytest`) das partes puras do sistema

**Algoritmo / precisão:**
- Suavização trocada de média móvel simples para EMA (menos atraso
  perceptível, mesmo efeito de estabilização)
- Confirmação de seleção por **piscada** (EAR - Eye Aspect Ratio) como
  alternativa ao dwell time (olhar fixo), configurável e combinável
- Compensação de **pose de cabeça** (yaw) usando a matriz de
  transformação facial do MediaPipe, para não confundir cabeça virada
  com movimento real dos olhos
- Cálculo de `ratio_y` (eixo vertical) disponível em `gaze_engine.py`,
  abrindo caminho para uma grade 2D / teclado por varredura no futuro
- Timeout de rosto não detectado, com aviso visual na tela
