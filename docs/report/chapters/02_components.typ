#import "@preview/supercharged-hm:0.1.0": *

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

#include("components/01_comp_user_interface.typ")
#include("components/02_comp_supervisor.typ")
#include("components/03_comp_file_server.typ")
#include("components/04_comp_text_extractor.typ")
#include("components/05_comp_layout_detector.typ")
#include("components/06_comp_font_detector.typ")
#include("components/07_comp_document_translator.typ")
#include("components/08_comp_document_class_detector.typ")
#include("components/09_comp_x_document_editor.typ")
#include("components/10_comp_document_image_renderer.typ")
