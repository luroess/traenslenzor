#import "@preview/supercharged-hm:0.1.2": *

= Components <components>

#figure(caption: [Architecture overview of components and established communication paths])[
  // the drawio source file can be found under ./docs/report/graphics/adl-report.drawio
  #image("../imgs/architecturev1.png")
] <components_architecture_fig>
All components of the image translator are visualized in @components_architecture_fig.
\
The user interacts through the web-based streamlit user interface described in @comp_user_interface.
There, the user can upload an image to be translated and specify a prompt with requests for the target language and further alterations of the result.
Upon the user starting the user interface, a new session is created via an #gls("http") #gls("api") call to the file server component, instantiating a session state with a unique session id.
\
The file server component described in @comp_file_server serves as the central storage location for all data associated with a session, including image data like the uploaded document and associated metadata such as extracted text and corresponding bounding boxes.
All data exchange between different tool components, the ui, and the supervisor is orchestrated via the file server component.
Once the user sends a prompt in the user interface, the supervisor component described in @comp_supervisor is invoked.
\
The supervisor is the central node of the program, running the agent #gls("llm") and is connected to all tool components.
Utilizing langchain as the framework for agentic #gls("llm") tasks, all tools are provided to the `gwen3:4b` #gls("llm") model.
Beginning with the first prompt received by the #gls("llm") in a session, the #gls("llm") will then deduce a logical order for tool execution from the required data dependencies and descriptions provided by the tools.
A fixed order is not provided.
In case the #gls("llm") requires further user input such as the destination language, it might respond to the user with a follow-up question for clarification.
\
Typically, the text extraction tool described in @comp_text_extractor is called as the first step.
First, the image borders are recognized and it is deskewed to a fitting rectangle. Then, utilizing an #gls("ocr") library, the text features and their locations are extracted from the image.
The resulting deskewed image and metadata are then uploaded to the file server.
\
Following text extraction, the #gls("llm") typically invokes the Font Detector (@comp_font_detector) and Document Translator (@comp_document_translator) to enrich each text item with font information and translations.
Once all items are fully processed, the Document Image Renderer (@comp_document_image_renderer) combines these results to produce the final translated document image.
The renderer uses #gls("ffc")-powered inpainting to remove the original text and places the translated text with matching fonts, preserving the document's original appearance.
\


#warning-note()[
  #strong[For each component:] Add a description (task, input, output, #emph[interesting] technical detail).
  For some components (e.g., the font detector or the document class detector), add experiment results (confusion matrices, loss curves, etc.).
]

#include "components/01_comp_user_interface.typ"
#include "components/02_comp_supervisor.typ"
#include "components/03_comp_file_server.typ"
#include "components/05_comp_document_scanner.typ"
#include "components/04_comp_text_extractor.typ"
#include "components/06_comp_font_detector.typ"
#include "components/07_comp_document_translator.typ"
#include "components/08_comp_document_class_detector.typ"
#include "components/09_comp_x_document_editor.typ"
#include "components/10_comp_document_image_renderer.typ"
