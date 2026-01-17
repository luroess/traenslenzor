#import "@preview/supercharged-hm:0.1.1": *
#import "@preview/pintorita:0.1.4"
#show raw.where(lang: "pintora"): it => pintorita.render(it.text, style: "default")

#let contributed(component, ..contributions) = [
  #figure(caption: [Student contributions linked to #component])[
  #table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Student*],[*WP ID*],[*Contribution Details*]),
    ..contributions
  )
  ]
]

= Team <team>

== Reflexion <team_reflexion>

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Reflect on your work in this project (provide 3–5 bullet points each, team effort).
]

#strong[What went right:]

- Lorem Ipsum

#strong[What went wrong:]

- Lorem Ipsum

== Work Packages <team_work_packages>

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Describe the work packages you've defined within the project.
]

#figure(caption: [Defined work packages])[
  #table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*WP ID*],[*Work Package Description*]),
    [UI1],[Basic UI],
    
    [SV1],[Experimentations with technologies.],
    [SV2],[Supervisor Setup.],
    [SV3],[Mock Infrastructure.],
    [SV4],[Evaluate different llms.],
    [SV5],[Multiple Tool Calls LLAMA 3.],
    [SV6],[Memory],
    [SV7],[Session Changes],
    [SV10],[Bug Fixes],

    [FS1],[File Server],
    [FS2],[Session Server],

    [TE1],[Flatten Image],
    [TE2],[Paddle OCR],
    [TE2],[Tesseract],


    [XDE1], [Direct Version],
    [XDE2], [Separate LLM],

  )
] <team_work_packages_work_packages_table>

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Describe the actual work on each work package per person.
]

#contributed(
  "UI",
  [Felix Schladt\ Jan Schaible],[UI1],[
    - Setup Streamlit UI
    - Basic Chat Interface to interact with the Supervisor.
  ]
)

#contributed(
  "Supervisor",
  [Felix Schladt\ Jan Schaible],[SV1],[
    - Langchain vs Langgraph
    - Compare different methods for tool calling
    - Investigate how Information could be passed between the tools.
  ],
  [Felix Schladt\ Jan Schaible],[SV2],[
    - Configure Langchain to use ollama.
    - Use ollama api to automatically pull model if not present
    - Setup read-eval-print loop to interact with the llm
    - Setup llm to use tools.
  ],
  [Felix Schladt\ Jan Schaible],[SV3],[
    - Create a mock Server for every tool.
    - Mock every tool so that it responds with sample data.
    - Convert app to async to work around the python GIL @pep703
  ],
  [Felix Schladt], [SV4], [
    // todo felix
  ],
  [Jan Schaible], [SV5], [
    - Investigating why LLAMA 3 fails to call multiple tools back to back.
    - Extend OLLAMA with a custom model that derives from LLAMA 3 with extended template to call tools back to back.
    - discarded as LLAMA3 was not capable to work reliably.
  ],
  [Felix Schladt\ Jan Schaible], [SV6], [
    - Create a tool for the llm to call so it can store important memory
    - inject that memory into the prompt so that the llm has it freshly in the context.
    - discarded as the llm failed to reliably store information that were relevant for later usage
  ],
  [Felix Schladt\ Jan Schaible], [SV7], [
    - Implement Changes Resulting from different Session concept.
    - Inject the session into the supervisor System prompt
  ],
  [Felix Schladt\ Jan Schaible], [SV10], [
    - Optimize Prompt
    - Minor Bug fixes
  ],
)

#contributed(
  "File Server",
  [Felix Schladt\ Jan Schaible], [FS1], [
    - Implementation of a in Memory HTTP File Server and Client.
  ],
  [Felix Schladt\ Jan Schaible], [FS2], [
    - Implementation of a in Memory HTTP Session Server and Client.
  ],
)

#contributed(
  "Text Extractor",
  [Felix Schladt], [TE1], [
    - Grayscale, blur, and detect edges with Canny.
  ],
  [Felix Schladt], [TE2], [
    - Perform OCR using Paddle.
    - Parse results into session data structure.
  ],
  [Jan Schaible], [TE3], [
    - Perform OCR using Tesseract.
    - Parse results into session data structure.
  ],
)

#contributed(
  "Document Editor",
  [Felix Schladt\ Jan Schaible], [XDE1], [
    - Implement a tool by which the supervisor can modify the text in the session.
    - Implement a tool by which the supervisor can get the current text from the session.
  ],
  [Felix Schladt\ Jan Schaible], [XDE2], [
    - Implement a tool which optimizes the text currently stored in the session accordiging to user instructions
    - Implement parsing
    - Implmenet Extra llm call
    - Implmenet mcp server for tool
  ],
)

#pagebreak()

== Timeline <team_timeline>

#figure(caption: [Project timeline by work package])[
```pintora
gantt
  dateFormat YYYY-MM-DDTHH
  axisFormat DD/MM
  axisInterval 2w

  section user_interface
  "Basic UI"       : 2026-1-10, 1w

  section supervisor
  "Experimentations with technologies"       : sv1, 2025-7-10, 2w
  "Supervisor Setup"       : sv2, after sv1, 4w
  "Mock Infrastructure"       : sv3, after sv2, 3w
  "Evaluate different llms"       : sv4, after sv3, 2w
  "Multiple Tool Calls LLAMA 3"       : sv5, after sv3, 3w
  "Memory"       : sv6, after sv5, 4w
  "Session Changes"       : sv7, 2025-11-5, 2w
  "Bug Fixes"       : sv10, 2025-12-20, 2w

  section file_server
  "File Server"       : fs1, after sv4, 1w
  "Session Server"       : fs2, after sv6, 1w

  section text_extractor
  "Flatten Image"       : te1, after sv6, 1w
  "Paddle OCR"       : te2, after te1, 1w
  "Tesseract"       : te3, 2026-1-1, 1w

  section layout_detector
  "Tool Mock"       : 2025-7-10, 1w

  section font_detector
  "Tool Mock"       : 2025-7-10, 1w

  section document_translator
  "Tool Mock"       : 2025-7-10, 1w

  section document_class_detector
  "Tool Mock"       : 2025-7-10, 1w

  section x_document_editor
  "Direct Version"       : xde1, 2025-11-25, 1w
  "Separate LLM"       : xde2, after xde1, 2w

  section document_image_renderer
  "Tool Mock"       : 2025-7-10, 1w

  section Release
  "Release" : milestone, 2026-1-23, 0d
```
] <team_timeline_gantt>
