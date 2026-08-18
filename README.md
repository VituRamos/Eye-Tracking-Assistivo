# Eye-Tracking Assistivo V2.0 👁️‍🗨️

Um sistema de rastreamento ocular de alta precisão desenvolvido em **Python** para fins assistivos (Comunicação Alternativa baseada em escolhas de "SIM / NÃO"), utilizando **MediaPipe Face Landmarker (Tasks API)** para a detecção de íris/olhos em tempo real e uma interface moderna construída com **CustomTkinter**.

---

## 🚀 Funcionalidades
- **Rastreamento de Olhar (Gaze Tracking):** Monitora a posição horizontal da íris em relação aos cantos dos olhos com estabilização por média móvel (redução de tremor).
- **Calibração Dinâmica Personalizada:** Sistema guiado de calibração em duas fases (Olhar para o SIM e olhar para o NÃO) para calibrar a sensibilidade de acordo com a amplitude de movimento ocular de cada utilizador.
- **Interface Moderna (Dark Mode):** Desenvolvida com CustomTkinter, oferecendo uma experiência visual limpa, ergonómica e em tela cheia (*Fullscreen*).
- **Seleção por Fixação Temporal:** Permite selecionar uma opção apenas mantendo o olhar fixo nela por um determinado período (ex: ~0.7 segundos), acompanhado por uma barra de progresso visual.

---

## 🛠️ Tecnologias e Dependências
O projeto foi construído utilizando as seguintes bibliotecas em Python:
- **Python 3.10+**
- `customtkinter` (Interface gráfica moderna)
- `opencv-python` (Processamento de vídeo e computação gráfica)
- `mediapipe` (Modelo de IA *Face Landmarker*)
- `numpy` (Cálculos matemáticos e manipulação de arrays)
- `pillow` (Manipulação e conversão de imagens para a UI)

---

## 📦 Instalação e Configuração

### 1. Clonar o repositório ou descarregar os ficheiros
Certifique-se de que se encontra na pasta raiz do projeto.

### 2. Instalar as Dependências
Execute o seguinte comando no seu terminal para instalar todos os pacotes necessários:
```bash
pip install customtkinter opencv-python mediapipe numpy pillow
