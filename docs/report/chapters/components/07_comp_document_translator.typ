#import "@preview/supercharged-hm:0.1.2": *

== Document Translator <comp_document_translator>

The Document Translator component transforms #gls("ocr")-extracted text items into the target language using the configured Ollama model (see @sec-llm-config and `settings.llm.model`).
The component is exposed as an #gls("mcp") tool through FastMCP, allowing the Supervisor to invoke translation on demand for a given session.

=== Batch Translation Strategy

Translating text items individually would result in several problems:
- *Loss of context:* Adjacent text fragments often share subject matter, terminology, or narrative flow. An #gls("llm") uses this cross-item context to select appropriate translations and maintain consistency.
- *Sequential bottleneck:* Ollama does not parallelize well, meaning per-item requests execute sequentially, making large documents unfeasible to translate in reasonable time.

To address these issues, the translator implements a batch-first approach:
- All `TextItem.extractedText` values from a session are collected and enumerated in the format `0: ...`, `1: ...`, etc.
- A single `client.chat` call is made with a system prompt instructing the #gls("llm") to preserve the original numbering and structure in the output.
- The response is parsed using regex to match indices, ensuring robustness even if the model returns lines out of order.
- If specific items are missing from the batch response, they are automatically retried individually.
- Each result is wrapped in a `TranslationInfo` object and attached via `add_translation`, producing a list of `HasTranslation` items.

=== Fallback to Sequential Translation

#figure(caption: [Batch translation with numbered format for robust parsing: `./traenslenzor/translator/translator.py:43-106`])[
    #code()[```py
    system = {
        "role": "system",
        "content": f"""
            You are an expert translator. Translate the following list
            of texts into target language: '{lang}'.

            ...

            Output must follow the same format as the input:
                1: Translated line 1
                2: Translated line 2

            Do not output anything other than the translated text.
        """,
    }

    message = {
        "role": "user",
        "content": "\n".join([f"{i}: {txt.replace(chr(10), ' ')}" for i, txt in enumerate(input_texts)]),
    }
    ```]
]<translator_batch_format>

The translator employs a robust two-stage fallback mechanism.
First, if the batch response is successfully parsed but some indices are missing, only those specific items are retried individually using the `translate()` function.
Second, if the batch request fails entirely (e.g., connection error or malformed non-parseable output), a global fallback triggers to translate every item sequentially.
This ensures robustness: even if the #gls("llm") fails to maintain structure or drops lines in batch mode, every item still receives a translation.
However, this comes at the cost of execution speed, as the sequential processing of many small requests is significantly slower than a single batch operation.

The fallback function uses a simpler single-item prompt that instructs the #gls("llm") to return only the translation or, if translation is impossible, the original text unchanged.

=== MCP Integration

The translator exposes a single tool via FastMCP:
- `translate(session_id: str)`: Retrieves the session from the File Server, performs batch translation on `session.text` into `session.language`, updates the session with the translated items, and returns a success message.

This design allows the Supervisor to invoke translation as a black-box operation without needing to manage translation logic or #gls("llm") calls directly.
