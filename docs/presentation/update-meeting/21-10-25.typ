#import "@preview/bytefield:0.0.6": *
#import "@preview/cetz:0.2.2"
#import "../static/themes.typ": *
#import "../static/notes.typ": *

#let code(body, highlighted: (), numbering: "1") = {
  set par(justify: false)
  sourcecode(highlighted: highlighted, numbering: numbering)[
    #body
  ]
}

//Red
#let hm_red = rgb("#fb5555")
#let hm_red_light = hm_red.lighten(40%)
#let hm_red_very_light = hm_red.lighten(75%)

// Black and White
#let hm_black = rgb("#222222")
#let hm_white = rgb("#FFFFFF")

// Grey
#let hm_grey_dark = rgb("#575756")
#let hm_grey_medium = rgb("#888888")
#let hm_grey_light = rgb("#B2B2B2")
#let hm_grey_very_light = rgb("#DEDFE0")

// Green
#let hm_green_dark = rgb("#13A256")
#let hm_green_medium = rgb("#50B264")
#let hm_green_light = rgb("#95C994")
#let hm_green_very_light = rgb("#95C994").lighten(60%)

#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

#let styledtable(
  stroke: hm_grey_dark,
  background_odd: hm_red_very_light,
  background_even: hm_white,
  _table_) = {
  set table.hline(stroke: stroke)
  set block(breakable: true)
  set table(
   align: left,
   fill: (_, y) => if calc.odd(y) {background_odd} else {background_even},
   stroke: none,
  )

  align(center,
    block(
      radius: 5pt,
      stroke: 0pt,
      clip: true,
      width: auto,
      breakable: true,
      _table_
    )
  )
}

#show: metropolis-theme.with(
  footer: [Group 12 - Advanced Deep Learning]
)

#set text(font: "Fira Sans", weight: "light", size: 18pt)
#show math.equation: set text(font: "Fira Math")
#set strong(delta: 100)
#set par(justify: true)

#let ncode(body) = {
  sourcecode(
    numbers-side: left,
    numbering: "1",
    numbers-start: 1,
    numbers-first: 1,
    numbers-step: 1,
    numbers-style: (i) => align(right, text(fill:blue, emph(i)))
  )[#body]
}

#let pos() = {
  set text(fill: lime.darken(30%), size: 20pt)
  [*+*]
}

#let neg() = {
  set text(fill: red, size: 20pt)
  [*-*]
}

#let todo() = {
  text(size: 120pt)[#emoji.chicken.baby #text(fill: gradient.linear(..color.map.rainbow))[TUDÜ]]
}

#title-slide(
  author: [von Jan Philip Schaible, Felix Schladt, Jan Duchscherer, Benedikt Koehler und Lukas Roess],
  title: [Update Meeting - 21.10.2025],
  subtitle: "Advanced Deep Learning",
  date:  "21.10.2025",
  extra: none,
)

#slide(title: "Table of Contents")[
  #show: columns.with(2, gutter: 12pt)
  #metropolis-outline
]

#new-section-slide("Tasks Worked On")

#slide(title: [Tasks Worked On])[
== *Jan S.:*
- void

== *Felix:*
- void

== *Jan D.:*
- void

== *Benedikt:*
- void

== *Lukas:*
- Setup repository
- Created meeting slides
- Planned & created roadmap
]

#new-section-slide("Tasks Worked On")

#slide(title: [Tasks Worked On])[
== *Jan S.:*
- void

== *Felix:*
- void

== *Jan D.:*
- void

== *Benedikt:*
- void

== *Lukas:*
- Setup repository
- Setup meeting slides
- Planned & created roadmap
]

#new-section-slide("Problems Encountered")

#slide(title: [Problems Encountered])[
== *Jan S.:*
- void

== *Felix:*
- void

== *Jan D.:*
- void

== *Benedikt:*
- void

== *Lukas:*
- None
]

#new-section-slide("Plans for Next Week")

#slide(title: [Problems Encountered])[
== *Jan S.:*
- void

== *Felix:*
- void

== *Jan D.:*
- void

== *Benedikt:*
- void

== *Lukas:*
- Read about MLP architecture
- Create concept for document translator
]