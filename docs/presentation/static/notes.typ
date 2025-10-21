// Copyright 2024 Thomas Gingele https://github.com/B1TC0R3
// Copyright 2024 Felix Schladt https://github.com/FelixSchladt

#import "@preview/drafting:0.1.2": *

#let note_inset         = 1em
#let note_border_radius = 0.5em

#let note_info_border_color        = black
#let note_info_background_color    = gray.lighten(80%)
#let note_warning_border_color     = red
#let note_warning_background_color = orange.lighten(80%)
#let note_good_border_color        = green
#let note_good_background_color    = lime.lighten(80%)

#let note(content,
          width: auto,
          background: note_info_background_color,
          border: note_info_border_color,
          bold: true
          ) = {
  let weight = "light"
  if bold {
    weight = "semibold"
  }
  set text(black, weight: weight)
  inline-note(
    content,
    stroke: border,
    rect  : rect.with(
      inset : note_inset,
      radius: note_border_radius,
      fill  : background,
      width: width
    )
  )
}

#let warning-note(content, width: auto) = {
  set text(black, weight: "semibold")
  inline-note(
    content,
    stroke: note_warning_border_color,
    rect  : rect.with(
      inset : note_inset,
      radius: note_border_radius,
      fill  : note_warning_background_color,
      width: width
    ),
  )
}

#let good-note(content, width: auto) = {
  set text(black, weight: "semibold")
  inline-note(
    content,
    stroke: note_good_border_color,
    rect  : rect.with(
      inset : note_inset,
      radius: note_border_radius,
      fill  : note_good_background_color,
      width: width
    ),
  )
}

#let todo() = {
  set text(black)
  text(size: 120pt)[#emoji.chicken.baby #text(fill: gradient.linear(..color.map.rainbow))[TUDÜ]]
}