#import "@preview/supercharged-hm:0.1.2": *
#import "@preview/wrap-it:0.1.1": wrap-content

= Introduction <introduction>


#figure(
  image("/imgs/success3_cropped.png"),
  caption: [Example translation],
)<fig-example-translation>

#v(.5cm)
#wrap-content(
  align: top + right,
  column-gutter: 30pt,
  columns: (1fr, auto),
  [#figure(caption: [Straightened\ Image])[
    #image("/imgs/sample_straight.jpg", height: 17em)
  ]<fig-example-straight>],
  [
    The Translenzor Document Assistant can translate documents in place through a convenient chat-based user interface.
    It is modeled after similar applications such as Google Translate.
    Users can upload images directly in the chat or load images from locations relative to the working directory.
    The system detects text blocks, translates them into another language, and preserves the original font, size, position, and orientation.
    In addition, users can modify text within the image via chat input, and the Supervisor (#gls("llm")) incorporates these changes, re-rendering the image if necessary.
    The system also recognizes document types, producing output that maintains the structure and appearance of the original, as illustrated in @fig-example-translation and @fig-example-straight.
  ],
)


