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

== Reflection <team_reflexion>

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
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*WP ID*],[*Work Package Description*]),
    [UI1],[Basic UI],
    
    [SV1],[Technology experiments.],
    [SV2],[Supervisor setup.],
    [SV3],[Mock infrastructure.],
    [SV4],[Evaluate different LLMs.],
    [SV5],[Multiple tool calls (LLaMA 3).],
    [SV6],[Memory.],
    [SV7],[Session changes.],
    [SV10],[Bug fixes.],

    [FS1],[File Server],
    [FS2],[Session Server],

    [TE1],[Flatten Image],
    [TE2],[Paddle OCR],
    [TE3],[Tesseract],


    [XDE1],[Direct Version],
    [XDE2],[Separate LLM],

  )
] <team_work_packages_work_packages_table>

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Describe the actual work on each work package per person.
]

#contributed(
  "UI",
  [Felix Schladt\ Jan Schaible],[UI1],[
    - Set up the Streamlit UI.
    - Built a basic chat interface to interact with the supervisor.
  ]
)

#contributed(
  "Supervisor",
  [Felix Schladt\ Jan Schaible],[SV1],[
    - Compared LangChain vs. LangGraph.
    - Compared different methods for tool calling.
    - Investigated how information could be passed between tools.
  ],
  [Felix Schladt\ Jan Schaible],[SV2],[
    - Configured LangChain to use Ollama.
    - Used the Ollama API to automatically pull models if not present.
    - Set up a read-eval-print loop (REPL) to interact with the LLM.
    - Enabled tool calling in the LLM.
  ],
  [Felix Schladt\ Jan Schaible],[SV3],[
    - Created a mock server for each tool.
    - Mocked each tool to return sample data.
    - Converted the app to async to work around the Python GIL @pep703.
  ],
  [Felix Schladt], [SV4], [
    // todo felix
  ],
  [Jan Schaible], [SV5], [
    - Investigated why LLaMA 3 fails to call multiple tools back-to-back.
    - Extended Ollama with a custom model derived from LLaMA 3 and an updated template for back-to-back tool calls.
    - Discarded because LLaMA 3 was not reliable enough.
  ],
  [Felix Schladt\ Jan Schaible], [SV6], [
    - Created a tool the LLM can call to persist important information.
    - Injected that memory into the prompt so the LLM has it in-context.
    - Discarded because the LLM did not reliably store information relevant for later use.
  ],
  [Felix Schladt\ Jan Schaible], [SV7], [
    - Implemented changes based on the updated session concept.
    - Injected the session into the supervisor system prompt.
  ],
  [Felix Schladt\ Jan Schaible], [SV10], [
    - Optimized prompts.
    - Fixed minor bugs.
  ],
)

#contributed(
  "File Server",
  [Felix Schladt\ Jan Schaible], [FS1], [
    - Implemented an in-memory HTTP file server and client.
  ],
  [Felix Schladt\ Jan Schaible], [FS2], [
    - Implemented an in-memory HTTP session server and client.
  ],
)

#contributed(
  "Text Extractor",
  [Felix Schladt], [TE1], [
    - Converted to grayscale, applied blurring, and detected edges with Canny.
  ],
  [Felix Schladt], [TE2], [
    - Performed OCR using Paddle.
    - Parsed results into the session data structure.
  ],
  [Jan Schaible], [TE3], [
    - Performed OCR using Tesseract.
    - Parsed results into the session data structure.
  ],
)

#contributed(
  "Document Editor",
  [Felix Schladt\ Jan Schaible], [XDE1], [
    - Implemented a tool for the supervisor to modify session text.
    - Implemented a tool for the supervisor to retrieve the current session text.
  ],
  [Felix Schladt\ Jan Schaible], [XDE2], [
    - Implemented a tool to optimize the text stored in the session according to user instructions.
    - Implemented parsing.
    - Implemented an additional LLM call.
    - Implemented an MCP server for the tool.
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
  "Technology experiments"       : sv1, 2025-7-10, 2w
  "Supervisor setup"       : sv2, after sv1, 4w
  "Mock Infrastructure"       : sv3, after sv2, 3w
  "Evaluate different LLMs"       : sv4, after sv3, 2w
  "Multiple tool calls (LLaMA 3)"       : sv5, after sv3, 3w
  "Memory"       : sv6, after sv5, 4w
  "Session changes"       : sv7, 2025-11-5, 2w
  "Bug fixes"       : sv10, 2025-12-20, 2w

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
