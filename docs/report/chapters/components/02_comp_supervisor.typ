#import "@preview/supercharged-hm:0.1.1": *

== Supervisor <comp_supervisor>

- Internal working
- Used model etc

The supervisor is the central component of the application. It is responsible for the interaction with the agent #gls("llm"), provides the #gls("llm") with callable tools and manages state.

=== Internal Structure 
- how session state is provided and structured
- langchain usage

=== Model Selection
#let model(m) = {rgb-raw(m, rgb("#5079ba"))}

Choosing a suitable #gls("llm") was challenging, as the system had to run on consumer hardware without #gls("gpu") support, restricting us to smaller models. This was further complicated by the requirement that the model determine the tool invocation order dynamically at runtime using only tool descriptions and input specifications.
Development started using the #model("gemma3:4b")@noauthor_gemma34b_nodate #gls("llm") model, which soon proved to be incompatible due to missing tool-calling abilities @schmid_google_2025.
Instead, #model("llama3.1:8b")@noauthor_llama31_nodate was chosen, offering good langchain integration and tool-calling abilities. 
During later development, however, it became apparent that the model could not reliably select the correct tools for execution and had difficulty following its instructions.
A short lived switch to #model("llama3.2:3b")@noauthor_llama32_nodate proved it to also not be up to the task, though coping better in some aspects.
To identify a suitable model capable of reliably handling user input and correctly selecting tools, a broad range of freely available models was evaluated.
This included #model("gpt-oss:20b")@noauthor_gpt-oss_nodate, #model("gwen3:8b")@noauthor_qwen3_nodate, #model("qwen3:14b"), #model("deepseek-r1:8b")@noauthor_deepseek-r1_nodate and #model("deepseek-r1:14b").
Though #model("gpt-oss:20b") proved very reliable and accurate, it proved quite resource hungry and usable on our development hardware.
Instead, #model("gwen3:8b") demonstrated good reasoning capability and reliably identified the correct oder of tools to call. With further testing and prompt refinement a step down to #model("gwen3:4b") also proved to work reliably.
Although #model("gwen3:4b") is comparatively small, its strong reasoning capabilities provide a high level of understanding, albeit at the cost of relatively long response times on our development systems.

=== Tool Calling

- mention set language tool
- how other tools are called

=== Prompt

- the llm must figure out the order in which to call tools from by itself
- tool descriptions are provided by langchain in addition to the prompt
- LLM should do as much as it can by itself but if required ask back to the user

#figure(caption: [Aggregated prompt template for #gls("llm") calls by the supervisor: `./traenslenzor/supervisor/prompt.py`])[
    #code()[```py
    f"""
    Task:
        You are an image translation assistant.

        Your task is to:
        - Translate all visible text in the provided image from the source language into the target language.
        - Produce a corresponding image with the translated text accurately placed.
        - The user might want to translate multiple documents.

        Tool usage:
        - If multiple tools are available, determine the correct execution order based on tool input/output dependencies.
        - Invoke a tool only when all required parameters are available.
        - Do not describe internal reasoning, planning, or tool usage.

        Missing information:
        - If required information (e.g., target language or image) is missing, ask the user a single concise clarifying question before proceeding.

        Output requirements:
        - After completing the image render for the first time, state the document type represented by the image.
        - Ask if the user would like to change something.

        User feedback to the rendered image:
        - If the user provides feedback, treat the input as exact argument for the “apply user feedback” tool.
        - Immediately call the “apply user feedback” tool with that content, without additional commentary.

    Context:
        {formatted_session}
    """
    ```]
]<supervisor_prompt>

