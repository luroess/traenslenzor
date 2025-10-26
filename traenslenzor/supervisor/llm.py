import requests
from langchain_ollama import ChatOllama

try:
    # check ollama server
    requests.get("http://localhost:11434", timeout=2)
except Exception:
    print("Error: Ollama server not running")
    exit(-1)

llm = ChatOllama(model="llama3.1", temperature=0, seed=69)
