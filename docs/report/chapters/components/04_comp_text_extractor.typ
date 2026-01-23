#import "@preview/supercharged-hm:0.1.2": *
#import "@preview/wrap-it:0.1.1": wrap-content

== Text Extractor (Layout Detector) <comp_text_extractor>

// - Show trial with Paddle OCr
//   - Explain issues
//     - Wrong text boxes
//     - Pull Request Logging
//     - Issues with langchain integration

// - Show pytesseract solution
//   - why its better an why tesseract was chosen
//   - etc

// - SHOW Preprocessing ( In his template Preprocessing is done in the Image Provider, which we do not really have. We have the file server doing something similar i guess but its definetely not the place where we do image preprocessing. I think its fine though, he said we do not have to follow the template 100%.)

#wrap-content(
  align: top+right,
  column-gutter: 30pt,
  columns: (1fr, 1fr),
  [#figure(caption: [#gls("ocr") test showcasing the detected text boxes and text.])[
    #image("/imgs/ocr_test_boxes.png")
  ]<fig-ocr-result>],
  [
The Text Extractor component is responsible for extracting the text and the areas where the text is contained in the document.
For this, a #gls("ocr") library is utilized.

== Document Deskew
If the session already contains an `ExtractedDocument` produced by the doc scanner (@comp_doc_scanner), the Text Extractor can directly run OCR on the deskewed image.
Otherwise, it falls back to a heuristic corner-based deskew described below.
The goal of document deskewing is to find a document in an image and straighten it so it looks like it was scanned from the top.
First, the image is converted to grayscale and slightly blurred to remove noise.
Edges are then detected using an edge detector, and a small morphological operation is applied to close gaps in the edges.
From this, the code finds contours in the image and looks at the largest ones, assuming the document is the biggest object in the picture.
Each contour is simplified, and if it has four corner points, is convex, and covers a large part of the image, it is treated as the document boundary.

After the document corners are found, they are put into a fixed order (top-left, top-right, bottom-right, bottom-left).
Then, we estimate the width and height of the document and compute a perspective transform.
This transform is used to warp the image so the document becomes rectangular and straight.
The final output is the deskewed image, along with the corner points and the transformation matrix, which can be used for #gls("ocr").

=== PaddleOCR

A first version of the text extractor was realized using PaddleOCR.
Integrating PaddleOCR into the application proved difficult due to it not being compatible with the used LangChain version 1.0.0.
As the incompatibility arose only due to two incorrect import paths in a single file, a small shim wrapper[@paddle_shim] requiring import prior to the PaddleOCR library import sufficed to fix the issue.


])

#wrap-content(
  align: top+right,
  columns: (2fr, 1.3fr),
  column-gutter: 30pt,
  figure(caption: [PaddleOCR detected text boxes (Image is shot from an angle and deskewed).])[
    #image("/imgs/paddle_ocr_boxes.png")
  ],
  [

  A further issue arose in conjunction with PaddleOCR.
  After running the text extraction tool call, log messages kept not showing anymore.
  The Paddle framework resets the application-wide log level configuration.
  The culprit could be identified and fixed through a pull request in the Paddle framework.
  Until the changes are downstreamed into PaddleOCR, the log level is reset after execution of the text extractor module@noauthor_paddlepaddlepaddle_nodate.

  Lastly, a more important and harder to solve problem arose with PaddleOCR.
  Though text recognition worked well, the bounding boxes reported by PaddleOCR proved to be inaccurate.
  This led to varied results when rendering the final image with the text renderer.
  This issue, also experienced by other groups in the lecture, proved not to be easily fixable.
  Therefore, the decision was made to integrate a different #gls("ocr") library with hopefully better results.

  === PyTesseract
  PyTesseract was selected as a replacement for PaddleOCR based on a comparison of open-source OCR libraries @ocr-comparison.
  The detection quality meets the requirements of our use case and produces the results shown in @fig-ocr-result.
  A key challenge was adapting the OCR output to our internal data structures.
  Since Tesseract provides word-level results, we aggregate words into line-level text items by grouping them by page, block, paragraph, and line identifiers, and further splitting them when large horizontal gaps or vertical misalignment indicate separate columns or snippets.
  Each resulting segment is merged into a single OCRTextItem with concatenated text, an averaged confidence score, and a bounding box enclosing all included words.

  ]
)

