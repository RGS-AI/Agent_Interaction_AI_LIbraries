# AI Agent App

A lightweight desktop AI agent powered by LangChain, Streamlit, and local LLMs (via Ollama), designed to:

- Detect installed AI/ML/DL libraries
- Perform system-level version checks
- Install missing packages after user confirmation
- Check GPU support for TensorFlow & PyTorch
- Maintain persistent chat history
- Log actions to a local SQLite database
- Run as a desktop app via PyInstaller

---

## Features

1. Cross-platform: macOS, Windows, Linux  
2. Local LLM via Ollama (Mistral by default)  
3. Streamlit-based UI  
4. Persistent chat & logs  
5. CLI and GUI support  
6. GPU support check (CUDA, MPS)  
7. PyInstaller `.spec` for packaging  
8. SQLite log viewer in-app  

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and `mistral` model pulled
- `pip install -r requirements.txt`

---

## Installation

### 1. Clone the repo

- git clone [https://github.com/yourusername/ai-agent-app.git](https://github.com/RGS-AI/Agent_Interaction_AI_LIbraries.git)
- cd ai-agent-app

## Pull and run Mistral Model with Ollama

- ollama pull mistral
- ollama run mistral

## Run the app
- streamlit run app.py

### Logging

- Logs are stored in agent_logs.db and can be viewed inside the app.

---

## Model Used
1. LLM: Mistral via Ollama
2. Agent Framework: LangChain
3. Interface: Streamlit

---

## Sample Screenshot

Coming soon

---

## License

MIT License

---

## Credits
Built by Raghunandan M S
