import logging
import os

import requests
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

SEED = 69
TEMPERATURE = 0

BASE_MODEL = "llama3.1"
MODEL_NAME = "traenslenzor_2000:0.1"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEMPLATE = """
{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}

{{ .System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities.
{{- end }}<|eot_id|>
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
{{- if and $.Tools $last }}

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{{ range $.Tools }}
{{- . }}
{{ end }}
Question: {{ .Content }}<|eot_id|>
{{- else }}

{{ .Content }}<|eot_id|>
{{- end }}{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
{{- if .ToolCalls }}
{{ range .ToolCalls }}
{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}{{ end }}
{{- else }}

{{ .Content }}
{{- end }}{{ if not $last }}<|eot_id|>{{ end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- end }}
{{- end }}
"""


try:
    # check ollama server
    requests.get(OLLAMA_URL, timeout=2)
except Exception:
    print("Error: Ollama server not running")
    exit(-1)


def pull_base_model():
    response = requests.post(f"{OLLAMA_URL}/api/pull", json={"model": BASE_MODEL})
    if response.status_code != 200:
        logger.error(f"failed to pull the base model '{BASE_MODEL}': \n{response.json()}")
        exit(-1)


def exists_model():
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    if response.status_code != 200:
        logger.error(f"failed to request model information: \n{response.json()}")
        exit(-1)
    present_models = response.json()["models"]
    model_names = [m["name"] for m in present_models]
    return MODEL_NAME in model_names


def delete():
    logger.info("Deleting model")
    response = requests.delete(f"{OLLAMA_URL}/api/delete", json={"model": MODEL_NAME})
    if response.status_code != 200:
        logger.error(f"failed to delete the model '{MODEL_NAME}': \n{response.json()}")
        exit(-1)


def create():
    logger.debug("Creating model")
    response = requests.post(
        f"{OLLAMA_URL}/api/create",
        json={"model": MODEL_NAME, "from": BASE_MODEL, "template": TEMPLATE},
    )
    if response.status_code != 200:
        logger.error(f"failed to create the model '{MODEL_NAME}': \n{response.json()}")
        exit(-1)


def initialize_model():
    if not exists_model():
        create()


initialize_model()
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    seed=SEED,
    base_url=OLLAMA_URL,
)
