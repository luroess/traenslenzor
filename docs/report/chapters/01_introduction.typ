#import "@preview/supercharged-hm:0.1.1": *
#import "@preview/wrap-it:0.1.1": wrap-content

= Introduction <introduction>


#figure(
  image("/imgs/success3_cropped.png"),
  caption: [Example translation]
)<fig-example-translation>

#v(.5cm)
#wrap-content(
  align: top+right,
  column-gutter: 30pt,
  columns: (1fr, auto),
  [#figure(caption: [Straightened\ Image])[
    #image("/imgs/sample_straight.jpg", height: 17em)
  ]<fig-example-straight>],
  [
    The Translänzor document assistant can translate multiple document images in succession by copying them directly into the chat, similar to existing tools.
    It can also load images from locations relative to its working directory.
    The system detects text blocks, translates them into another language, and preserves the original font, size, position, and orientation.
    In addition, users can modify text within the image via chat input, and the supervisor (#gls("llm")) incorporates these changes, rerendering the image if necessary.
    The system also recognizes document types, providing output that maintains the structure and appearance of the original, as illustrated in @fig-example-translation and @fig-example-straight.

  ]
)


