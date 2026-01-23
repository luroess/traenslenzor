#import "@preview/supercharged-hm:0.1.2": *
#show link: underline

== Supervisor <comp_supervisor>

The supervisor is the central component of the application.
It is responsible for interaction with the agent #gls("llm") and provides the #gls("llm") with callable tools.
Although the implementation is concise, it reflects extensive experimentation to achieve a reliable solution.

Substantial time was spent tuning the #gls("llm") prompt, and it remained one of the most sensitive parts of the system.
Even minor refinements intended to improve alignment could introduce subtle regressions, so each change required careful testing and multiple rounds of iteration.

=== Internal Structure
One of the requirements was to avoid programming a fixed sequence of tools that the process would follow once all information was gathered.
Therefore, we only use LanGraph indirectly via LangChain's create_agent method, which handles tool execution after an #gls("llm") call when specified.

To provide the current context, we leverage LangChain's `dynamic_prompt` hook to inject session context into the #gls("llm") (see @sec-prompt for details).
We opted to let the #gls("llm") inject the `session_id` into tool calls directly.
Programmatic injection would have required modifying the tool definitions, which we deemed unnecessary since the #gls("llm") handles the `session_id` injection seamlessly.

=== Model Selection <sec-llm-config>
#let model(m) = { rgb-raw(m, rgb("#5079ba")) }

Choosing a suitable #gls("llm") was challenging, as the system had to run on consumer hardware without #gls("gpu") support, restricting us to smaller models.
This was further complicated by the requirement that the model determine the tool invocation order dynamically at runtime using only tool descriptions and input specifications.
Development started using the #model("gemma3:4b")@noauthor_gemma34b_nodate #gls("llm") model, which soon proved to be incompatible due to missing tool-calling abilities @schmid_google_2025.
Instead, #model("llama3.1:8b")@noauthor_llama31_nodate was chosen, offering good LangChain integration and tool-calling abilities.
During later development, however, it became apparent that the model could not reliably select the correct tools for execution and had difficulty following its instructions.
A short-lived switch to #model("llama3.2:3b")@noauthor_llama32_nodate proved it to also not be up to the task, though coping better in some aspects.
To identify a suitable model capable of reliably handling user input and correctly selecting tools, a broad range of freely available models was evaluated.
This included #model("gpt-oss:20b")@noauthor_gpt-oss_nodate, #model("gwen3:8b")@noauthor_qwen3_nodate, #model("qwen3:14b"), #model("deepseek-r1:8b")@noauthor_deepseek-r1_nodate, and #model("deepseek-r1:14b").
Though #model("gpt-oss:20b") proved very reliable and accurate, it is also quite resource hungry and not usable on our development hardware.
Instead, #model("gwen3:8b") demonstrated good reasoning capability and reliably identified the correct order of tools to call.
With further testing and prompt refinement, a step down to #model("gwen3:4b") also proved to work reliably.
Although #model("gwen3:4b") is comparatively small, its strong reasoning capabilities provide a high level of understanding, albeit at the cost of relatively long response times on our development systems.

=== Prompt<sec-prompt>
One of the requirements was to avoid including an explicit task order.
Therefore, we structured the prompt to clearly define the task, tool usage rules, and how to handle missing information, while leaving the execution order up to the #gls("llm").
This system prompt is then injected, together with tool definitions and the most recent messages, into the prompt template @ollama_prompt_qwen.
The overall goal was to make the system as autonomous as possible while ensuring predictable, user-friendly behavior.

Some of the problems we encountered along the way included:

- *No rerender:* The #gls("llm") did not execute the render image tool after the user modified the text.
- *Imagined tools (LLaMA 3):* The #gls("llm") generated non-existent tools, e.g., ```py {"name": "language_selector"}```.
- *Simulated processing (LLaMA 3):* The #gls("llm") printed messages like `Rendering image...` instead of actually calling the tool.
- *Assumed language (LLaMA 3):* The #gls("llm") did not request the target language from the user when none had been provided and assumed a default.
- *Additional text (LLaMA 3):* The #gls("llm") added descriptive text around the actual tool call, e.g.:
  ```
  Extracting text from image...
  {"name": "text_extractor", "parameters": {}}
  ```
- This list is not exhaustive and only highlights some of the more frequent issues encountered.

#figure(
  caption: [Aggregated prompt template for #gls("llm") calls by the supervisor: `./traenslenzor/supervisor/prompt.py`],
)[
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
      ✅ the current session_id is '01329c88-0373-46df-a572-6c34e4a19c24'
      ✅ the user has selected the language english
      ❌ the user has no document selected"
      ...
  """
  ```]
]<supervisor_prompt>

=== Failed approaches

In this chapter, we discuss some approaches that were tried but later discarded.

==== Successive Tool Unlock

In the initial prototyping phase of the project, the key question was how a reliable tool call order could be ensured.
As no restrictions on how this should be achieved were specified in the initial project presentation and task description, we evaluated multiple variants to reliably achieve a fixed tool calling order.

Most successful, and well compatible with LangChain, proved to be the dynamic addition of tools as soon as the required prerequisites were met.
This addressed multiple issues we would otherwise face, including incorrect calling order by the #gls("llm") and incorrect call parameters, and it proved reliable in practice.
Using this method, a fully functional supervisor mockup was implemented in the initial phase of the project.

Once presented during the first update meeting, the requirements around the supervisor and tool calling were adjusted without prior notice in a way that made this solution no longer applicable.
As a result, we had to discard the working implementation and move to a different approach, which required additional effort and workarounds to reach comparable reliability.

==== LLaMA 3.1/3.2

As smaller models run faster and we aimed for a performant tool, we initially focused on using LLaMA 3.
Some of the challenges with this approach are discussed in @sec-prompt.
In particular, we observed that LLaMA 3 was unable to call multiple tools in succession.

Upon investigation, we found that the Ollama template used to construct the final prompt did not include tool descriptions when the last message was a response from a tool (see @ollama_prompt_llama).
We modified the template by deriving a custom model from LLaMA 3 using the Ollama interface at the beginning of the program.
Despite this, the #gls("llm") still failed to execute multiple tools sequentially.

The issue was traced to the sentence: "When you receive a tool call response, use the output to format an answer to the original user question."
Removing this instruction allowed the #gls("llm") to call tools successfully.
However, it then began calling tools continuously without stopping, which ultimately led us to switch to a different model.

==== Memory
As our first approach was to let the #gls("llm") handle the results returned from the tools directly, we needed some form of persistence that would keep relevant information in the context window even as the amount of data in the window grew.
To do this, we first considered approaches like #link("https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization")[LangChain's Summarization], but this might cut relevant information.
Therefore, we considered giving the #gls("llm") a tool to store relevant information in memory directly, which would then be injected into the context itself.
This, however, did not work and was one of the reasons we switched to the session-based system.

==== React pattern for tools
To encourage reasoning in LLaMA, we attempted to apply the ReAct pattern using a one-shot instruction:

#figure(
  caption: "One Shot Instruction",
  code()[```json
  Respond in the format {
      "reason": "reason you are calling this tool next",
      "name": function name,
      "parameters": dictionary of argument name and its value
  }...
  ```],
)
The goal was to prompt the model to explicitly provide a reason before invoking a tool.
However, LLaMA did not respond to this instruction, and its tool-calling behavior remained unchanged.
In contrast, with Qwen3 this measure was unnecessary, as the model inherently produces reasoning in a section preceding its output, which is then stripped by the Ollama interface.
