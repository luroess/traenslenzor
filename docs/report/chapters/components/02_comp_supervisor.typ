#import "@preview/supercharged-hm:0.1.1": *

== Supervisor <comp_supervisor>

- Internal working
- Used model etc

The supervisor is the central component of the application. It is responsible for the interaction with the agent #gls("llm"), provides the #gls("llm") with callable tools and manages state.

=== Internal Structure 
- how session state is provided and structured


=== Prompt

- the llm must figure out the order in which to call tools from by itself
- tool descriptions are provided by langchain in addition to the prompt
- LLM should do as much as it can by itself but if required ask back to the user


#figure(caption: [Aggregated prompt template for #gls("llm") calls by the supervisor: `./traenslenzor/supervisor/prompt.py`])[
    #code()[```py
    f"""
    Task:
        You are an image translation assistant.
        Your task is to translate all visible text in an image from the source language into the target language and produce a corresponding translated image.

        When multiple tools are available, determine the execution order based on the required inputs and outputs of each tool, ensuring that all required parameters are available before a tool is invoked.

        Do not describe internal reasoning, planned actions, or tool usage.

        If required information is missing (e.g. target language or document), ask the user a concise clarifying question before proceeding.

        After completing the translation, state the document type the image represents.

    Context:
        ✅ the current session_id is '{session_id}'
        {f"✅ the user has selected the language {session.language}"
            if session.language else 
          "❌ the user has no language selected"}
        {"✅ the user has a document loaded" 
            if session.rawDocumentId else 
         "❌ the user has no document selected"}
        {"✅ text was extracted from the document" 
            if has_text_been_extracted(session) else 
         "❌ no text was extracted from the document"}
        {"✅ the text was translated" 
            if has_translated_text(session) else 
         "❌ the text has not yet been translated"}
        {"✅ the font has been detected" 
            if has_font_been_detected(session) else 
         "❌ the font has not yet been detected"}
        {"✅ the document has been classified" 
            if has_document_been_classified(session) else 
         "❌ the document has not yet been classified"}
        {"✅ the result has been rendered" 
            if has_result_been_rendered(session) else 
         "❌ the result has not yet been rendered"}
    """
    ```]
]<supervisor_prompt>

=== Tool Calling

- mention set language tool
- how other tools are called

=== Model Selection

- intention
- llama3.1
- llama3.2
- Other models tried
- selection of qwen3:4b

