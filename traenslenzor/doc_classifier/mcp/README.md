# Doc Classifier MCP

## Endpoint
- Host/port: `127.0.0.1:8002`
- Base path: `http://127.0.0.1:8002/mcp` (exported as `DOC_CLASSIFIER_BASE_PATH`)
- Run locally:

```sh
PYTHONPATH=. .venv/bin/python - <<'PY'
import asyncio
from traenslenzor.doc_classifier.mcp.mcp_server import run
asyncio.run(run())
PY
```

## Tool signature

- **Input:** `path` (Path to image), `top_k` (1–16).
- **Output:** `probabilities` (`dict[label -> probability]`), `top_k`

## Examples

**Make a classification request:**

```sh
uv run python -m traenslenzor.doc_classifier.mcp.request
```
The output will be similar to:
```
{
    'probabilities': {
        'handwritten': 0.10379903791451665,
        'questionnaire': 0.09860097408998154,
        'email': 0.09134849964506403,
    },
    'top_k': 3,
}
```

**Inspect the tool's input/output schemas:**

```sh
uv run python -m traenslenzor.doc_classifier.mcp.get_schema
```
The output will be similar to:
```
Tool: 'classify_document'
{
    'input': None,
    'output': {
        'description': 'Structured response returned by the MCP tool.',
        'properties': {
            'probabilities': {
                'additionalProperties': {
                    'type': 'number',
                },
                'description': (
                    'Mapping of class name → probability (softmax scores for the returned top-k; values may not sum to'
                    ' 1 if top_k < num_classes).'
                ),
                'type': 'object',
            },
            'top_k': {
                'description': 'Number of classes included in the response.',
                'type': 'integer',
            },
            'model_version': {
                'anyOf': [
                    {
                        'type': 'string',
                    },
                    {
                        'type': 'null',
                    },
                ],
                'default': None,
                'description': 'Identifier for the underlying model/checkpoint.',
            },
        },
        'required': [
            'probabilities',
            'top_k',
        ],
        'type': 'object',
    },
}
```
