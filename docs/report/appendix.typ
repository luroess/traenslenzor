#import "@preview/supercharged-hm:0.1.1": *

#let appendix = [
// YEP another bug, but I will fix it later.... :/
#set heading(numbering: "a.")
#counter(heading).update((first, second) => (first - 2, second - 3))

== PaddleOCR Import Shim <paddle_shim> 

#figure(caption: [PaddleOCR Import Shim (replaced by pytesseract): `traenslenzor/text_extractor/shim_langchain_backcomp.py`])[
  #code()[```py
import sys
import types

# Ugly fix until paddle ocr is langchain 1.0.0 compatible
# https://github.com/PaddlePaddle/PaddleOCR/issues/16711#issuecomment-3446427004

# Provide old import paths expected by paddlex:
# langchain.docstore.document -> Document
m1 = types.ModuleType("langchain.docstore.document")
from langchain_core.documents import Document  # noqa: E402, I001

m1.Document = Document
sys.modules["langchain.docstore.document"] = m1

# langchain.text_splitter -> RecursiveCharacterTextSplitter
m2 = types.ModuleType("langchain.text_splitter")
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402, I001

m2.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter  # type: ignore
sys.modules["langchain.text_splitter"] = m2
```]
]


#figure(caption: [Promt template llama3.2])[
    #code()[```go
      <|start_header_id|>system<|end_header_id|>

      Cutting Knowledge Date: December 2023

      {{ if .System }}{{ .System }}
      {{- end }}
      {{- if .Tools }}When you receive a tool call response, use the output to format an answer to the orginal user question.

      You are a helpful assistant with tool calling capabilities.
      {{- end }}<|eot_id|>
      {{- range $i, $_ := .Messages }}
      {{- $last := eq (len (slice $.Messages $i)) 1 }}
      {{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
      {{- if and $.Tools $last }}

      Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

      Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

      {{ range $.Tools }}
      {{- . }}
      {{ end }}
      {{ .Content }}<|eot_id|>
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
```]
]<ollama_prompt_llama>

#figure(caption: [Promt template qwen3])[
    #code()[```go
    {{- $lastUserIdx := -1 -}}
    {{- range $idx, $msg := .Messages -}}
    {{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
    {{- end }}
    {{- if or .System .Tools }}<|im_start|>system
    {{ if .System }}{{ .System }}

    {{ end }}
    {{- if .Tools }}# Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {{- range .Tools }}
    {"type": "function", "function": {{ .Function }}}
    {{- end }}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>
    {{- end -}}
    <|im_end|>
    {{ end }}
    {{- range $i, $_ := .Messages }}
    {{- $last := eq (len (slice $.Messages $i)) 1 -}}
    {{- if eq .Role "user" }}<|im_start|>user
    {{ .Content }}<|im_end|>
    {{ else if eq .Role "assistant" }}<|im_start|>assistant
    {{ if (and $.IsThinkSet (and .Thinking (or $last (gt $i $lastUserIdx)))) -}}
    <think>{{ .Thinking }}</think>
    {{ end -}}
    {{ if .Content }}{{ .Content }}{{ end }}
    {{- if .ToolCalls }}
    {{- range .ToolCalls }}
    <tool_call>
    {"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
    </tool_call>
    {{- end }}
    {{- end }}{{ if not $last }}<|im_end|>
    {{ end }}
    {{- else if eq .Role "tool" }}<|im_start|>user
    <tool_response>
    {{ .Content }}
    </tool_response><|im_end|>
    {{ end }}
    {{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
    <think>
    {{ end }}
    {{- end }}
    ```]
]<ollama_prompt_qwen>

]
