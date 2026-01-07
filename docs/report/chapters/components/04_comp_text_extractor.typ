#import "@preview/supercharged-hm:0.1.0": *
#import "@preview/wrap-it:0.1.1": wrap-content

== Text Extractor <comp_text_extractor>

// - Show trial with Paddle OCr
//   - Explain issues
//     - Wrong text boxes
//     - Pull Request Logging
//     - Issues with langchain integration

// - Show pytesseract solution
//   - why its better an why tesseract was chosen
//   - etc

// - SHOW Preprocessing ( In his template Preprocessing is done in the Image Provider, which we do not really have. We have the file server doing something similar i guess but its definetely not the place where we do image preprocessing. I think its fine though, he said we do not have to follow the template 100%.)

#warning-note("Currently this only covers the ocr part, is rather problem and development history heavy. This should further include the outputted data (BBOXES Datastructure) the tooling endpoint etc")

#wrap-content(
  align: top+right,
  column-gutter: 30pt,
  columns: (1fr, 1fr),
  figure(caption: [#gls("ocr") test showcasing the detected text boxes and text.])[
    #image("/imgs/ocr_test_boxes.png")
  ],
  [
The Text Extractor component is responsible for extracting the text and the areas where the text is contained in the document.
For this a #gls("ocr") library is utilized.


=== PaddleOCR

A first version of the text extractor was realized using PaddleOCR.
Integrating PaddleOCR into the application proved difficult due to it not being compatible with the used Langchain version 1.0.0. 
As the incompatibility arose only due to two incorrect import paths in a single file, a small shim wrapper[@paddle_shim] requiring import prior to the PaddleOCR library import sufficed to fix the issue.

A further issue arose in conjunction with PaddleOCR. After running the text extraction tool call, log messages kept not showing anymore. 
The Paddle framework resets the application wide log level configuration. The culprit could be identified and fixed through a pull request in the Paddle framework. Until the changes would be downstreamed into PaddleOCR, the log level is reset after execution of the text extractor module@noauthor_paddlepaddlepaddle_nodate.
])

#wrap-content(
  align: top+right,
  columns: (2fr, 1.3fr),
  column-gutter: 30pt,
  figure(caption: [PaddleOCR detected text boxes (Image is shot from an angle and deskewed).])[
    #image("/imgs/paddle_ocr_boxes.png")
  ],
  [
Lastly, a more important and harder to solve problem arose with PaddleOCR.
Though text recognition worked well, the bounding boxes reported by PaddleOCR proved to be inaccurate. This lead to varied results when rendering the final image by the text renderer.
This issue, also experienced by other groups in the lecture, proved not to be easily fixable.
Therefore the decision was made to integrate a different #gls("ocr") library with hopefully better results.

=== PyTesseract
    TODO

    // Once your well below the image it may prove beneficial to move the following text underneath this block
  ]
)


