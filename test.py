# Script
import os
import platform
import subprocess
import streamlit as st
import sqlite3
from datetime import datetime
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory

# Chat History for Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Sqlite
def log_to_db(action: str, detail: str):
    conn = sqlite3.connect("agent_logs.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            detail TEXT
        )
    ''')
    cursor.execute("INSERT INTO logs (timestamp, action, detail) VALUES (?, ?, ?)",
                   (datetime.now().isoformat(), action, detail))
    conn.commit()
    conn.close()

# Having the list of libs for DL-AI to be installed, to be checked
required_libs = [
    'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
    'tensorflow', 'torch', 'xgboost', 'lightgbm', 'nltk', 'spacy',
    'transformers', 'datasets', 'opencv-python', 'keras'
]

def get_installed_packages():
    output = subprocess.check_output(["pip", "freeze"]).decode()
    return {pkg.split("==")[0].lower(): pkg.split("==")[1] for pkg in output.splitlines() if "==" in pkg}

def check_missing_packages(_: str) -> str:
    installed = get_installed_packages()
    missing = [pkg for pkg in required_libs if pkg.lower() not in installed]
    if not missing:
        return "All required AI/ML/DL libraries are already installed."
    return "Missing Libraries: " + ", ".join(missing)

def install_missing_packages(_: str) -> str:
    installed = get_installed_packages()
    missing = [pkg for pkg in required_libs if pkg.lower() not in installed]
    installed_list = []
    for pkg in missing:
        subprocess.run(["pip", "install", pkg])
        log_to_db("Install", f"Installed {pkg}")
        installed_list.append(pkg)
    return "Installed: " + ", ".join(installed_list) if installed_list else "No libraries to install."

# Checking the GPU availability
def check_gpu_support(_: str) -> str:
    try:
        import torch
        import tensorflow as tf
        torch_gpu = torch.cuda.is_available()
        tf_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        return f"PyTorch GPU: {'Available' if torch_gpu else 'Not Available'}\nTensorFlow GPU: {'Available' if tf_gpu else 'Not Available'}"
    except Exception as e:
        return f"Error during GPU check: {str(e)}"

# Using LangChain Tools 
tools = [
    Tool(name="CheckMissingLibraries", func=check_missing_packages, description="Check missing AI/ML/DL libraries"),
    Tool(name="InstallLibraries", func=install_missing_packages, description="Install missing AI/ML/DL libraries"),
    Tool(name="CheckGPU", func=check_gpu_support, description="Check GPU support for PyTorch and TensorFlow"),
]

llm = Ollama(model="mistral")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True, handle_parsing_errors=True)

# Streamlit UI
st.set_page_config(page_title="AI Agent CLI", layout="wide")
st.title("Cross-Platform AI Agent")

with st.expander("📋 Instructions"):
    st.markdown("""
        - This AI agent checks for missing AI/ML/DL libraries and installs them.
        - It also checks GPU availability.
        - All actions are logged to a local SQLite database.
    """)

query = st.text_input("💬 Ask me to check or install libraries, or check GPU support:")

if query:
    try:
        response = agent.run(query)
        st.success(response)
    except Exception as e:
        st.error(str(e))

# Chat History
with st.expander("💬 Chat History"):
    st.write(memory.buffer)

# Log Viewer
# Ensure logs table exists before querying
def ensure_logs_table():
    conn = sqlite3.connect("agent_logs.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            detail TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call this before querying logs
ensure_logs_table()

# Now fetch logs
conn = sqlite3.connect("agent_logs.db")
logs_df = conn.execute("SELECT * FROM logs ORDER BY timestamp DESC").fetchall()
conn.close()

with st.expander("Action Logs"):
    conn = sqlite3.connect("agent_logs.db")
    logs_df = conn.execute("SELECT * FROM logs ORDER BY timestamp DESC").fetchall()
    conn.close()
    if logs_df:
        import pandas as pd
        df = pd.DataFrame(logs_df, columns=["ID", "Timestamp", "Action", "Detail"])
        st.dataframe(df)
    else:
        st.info("No logs found.")