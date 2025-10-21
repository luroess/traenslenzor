#import "/template/lib.typ": *

// Minimal doc setup (optional)
#set page(paper: "a4", margin: 2cm)
#set heading(numbering: "1.")
#set text(size: 11pt)
#set par(justify: true)

= Components

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  #strong[To do.] Add a diagram of your architecture and explain it.
]

See @fig:arch for a diagram of the overall architecture.

#figure(caption: [Architecture])[
  #image("/imgs/arch.png", width: 80%)
] <fig:arch>

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  #strong[For each component:] Add a description (task, input, output, #emph[interesting] technical detail).
  For some components (e.g., the font detector or the document class detector), add experiment results (confusion matrices, loss curves, etc.).
]

== Supervisor

Einstein's famous equation relates energy and mass:

$ E = m c^2 $ <eq:einstein>

Equation @eq:einstein demonstrates the relation between energy and mass (see also @einstein1905).

== Image Provider

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Describe the image sources, formats, and any preprocessing (e.g., cropping, DPI normalization).
]

== Layout Detector

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Describe detection targets (text blocks, figures, tables), model/heuristics, inputs/outputs, and metrics.
]

== Font Detector

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Summarize classes, training data, features, model, and inference output.
  Include experiment results (e.g., confusion matrix, accuracy per class, example errors).
]

== Document Class Detector

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Define classes (report, homework, proposal, essay, slides, …), inputs/features, and decision logic.
  Add evaluation (precision/recall, ROC, confusion matrix).
]

== Document Translator

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Explain translation steps, language support, glossaries, and quality assurance.
]

== Document Image Renderer

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Describe rendering pipeline (fonts, pagination, resolution, export formats) and performance considerations.
]

== X

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Placeholder for an additional component. Describe task, I/O, method, and key detail.
]

= Team

== Reflexion

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Reflect on your work in this project (provide 3–5 bullet points each, team effort).
]

#strong[What went right:]

- Lorem Ipsum

#strong[What went wrong:]

- Lorem Ipsum

== Work Packages

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
    [WP1],[Lab report with methodology, data analysis, and conclusion.],
    [WP2],[Homework set requiring proofs, diagrams, and calculations.],
    [WP3],[Project proposal including scope, methods, and timeline.],
    [WP4],[Essay draft with thesis, arguments, and referenced sources.],
    [WP5],[Presentation slides with visuals, charts, and rehearsed flow]
  )
] <tab:work-packages>

#box(fill: luma(240), inset: 8pt, radius: 6pt)[
  Describe the actual work on each work package per person.
]

#figure(caption: [Student contributions linked to multiple work packages])[
  #table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Student*],[*WP ID*],[*Contribution Details*]),

    [Alice Johnson],[WP1],[Focused on error analysis and linked results back to the hypothesis],
    [Alice Johnson],[WP3],[Contributed background section and drafted methods],

    [Bob Smith],[WP2],[Produced detailed proofs and geometry diagrams],
    [Bob Smith],[WP4],[Assisted with logical flow and edited transitions],

    [Carol White],[WP3],[Led milestone planning and resource identification],
    [Carol White],[WP5],[Designed consistent slide templates and structure],

    [David Brown],[WP1],[Edited discussion and conclusion for clarity],
    [David Brown],[WP4],[Wrote introduction and thesis statement],

    [Eve Miller],[WP5],[Created visuals and rehearsed timing],
    [Eve Miller],[WP2],[Checked calculations for consistency]
  )
] <tab:student-work>

== Timeline

#figure(caption: [Project timeline by work package])[
  #box(fill: luma(240), inset: 8pt, radius: 6pt)[
    Add a chart here (x-axis: time, y-axis: work package).  
    Options:
    - Render an external chart to `images/timeline.png` and include via `#image()`.
    - Or create a simple Gantt-like table with start/end dates.
  ]
] <fig:timeline>
