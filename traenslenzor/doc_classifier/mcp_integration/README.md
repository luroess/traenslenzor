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


**Inspect the tool's input/output schemas:**

```sh
uv run python -m traenslenzor.doc_classifier.mcp.get_schema
```
The output will be similar to:
```
{
    'name': 'classify_document',
    'description': (
        'Classify a document image into one of the supported document classes. Provide the file id returned by FileCli'
        'ent.put_img.'
    ),
    'annotations': None,
    'input': {
        'properties': {
            'document_id': {
                'type': 'string',
            },
            'top_k': {
                'anyOf': [
                    {
                        'type': 'integer',
                    },
                    {
                        'type': 'string',
                    },
                ],
                'default': 3,
            },
        },
        'required': [
            'document_id',
        ],
        'type': 'object',
    },
    'output': {
        'description': 'Structured response returned by the MCP tool.',
        'properties': {
            'probabilities': {
                'additionalProperties': {
                    'type': 'number',
                },
                'description': 'Mapping of class name to probability for the top-k predicted classes.',
                'type': 'object',
            },
        },
        'required': [
            'probabilities',
        ],
        'type': 'object',
    },
}
```
