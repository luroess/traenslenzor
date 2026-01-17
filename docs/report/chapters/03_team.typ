#import "@preview/supercharged-hm:0.1.1": *

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
    [WP1],[Lab report with methodology, data analysis, and conclusion.],
    [WP2],[Homework set requiring proofs, diagrams, and calculations.],
    [WP3],[Project proposal including scope, methods, and timeline.],
    [WP4],[Essay draft with thesis, arguments, and referenced sources.],
    [WP5],[Presentation slides with visuals, charts, and rehearsed flow]
  )
] <team_work_packages_work_packages_table>

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
] <team_work_packages_contributions_table>

== Timeline <team_timeline>

#figure(caption: [Project timeline by work package])[
  #warning-note()[
    #set align(left)
    Add a chart here (x-axis: time, y-axis: work package).  
    Options:
    - Render an external chart to `images/timeline.png` and include via `#image()`.
    - Or create a simple Gantt-like table with start/end dates.
    - I have looked for gantt packages for typst.
      - Pintora
        - uses the pintora framework to render charts including gantt
        - looks quite simple and well documented (i would favor this)
        - https://typst.app/universe/package/pintorita/
        - https://pintorajs.vercel.app/docs/diagrams/gantt-diagram/
      - timeliney
        - https://typst.app/universe/package/timeliney
      - ganttty 
        - https://typst.app/universe/package/gantty/
  ]
] <team_timeline_gantt>
