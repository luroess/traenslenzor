#import "@preview/supercharged-hm:0.1.2": *

== X - Document Editor <comp_x_document_editor>

#figure(
  caption: "input, output to the document editor",
  box(width: 20em)[
    #code()[```yaml
    input:
      1: Text Beispuhl 1
      2: Text Beispuhl 1
    user_suggestion: "Fix the spelling"
    output:
      1: Text Beispiel 1
      2: Text Beispiel 1
    ```]
  ]
)<document-editor>

The Text Feedback Component allows users to provide suggestions for improving translated text in a session.
It combines the current session text with user input and sends it to a large language model via the Ollama client.
An example of this can be seen in @fig-example-translation.
The #gls("llm") generates an optimized version that incorporates the suggestions while preserving the original line structure.
In the first iteration, this functionality was implemented by letting the Supervisor handle the update directly using two additional tools: one to retrieve the current session text and another to update a specific line number with a replacement text.
In practice, this approach confused the Supervisor: it frequently replied with the corrected text instead of calling the update tool, and it never re-rendered the finished image after applying changes.
For parsing the response, a simple numbered line format (see @document-editor) was used instead of JSON, which proved more robust and reliable for preserving line order and handling minor formatting variations.
The component updates the session with the corrected text, enabling precise, line-by-line refinement.
Since the #gls("llm") cannot determine when the image should be re-rendered, the tool provides a message prompting the #gls("llm") to trigger image re-rendering when necessary.
