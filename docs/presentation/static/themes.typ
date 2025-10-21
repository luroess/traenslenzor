#import "@preview/polylux:0.3.1": *
#import "@preview/codelst:2.0.2": sourcecode


#let ncode(body, highlighted: (), numbering: "1") = {
  set par(justify: false)
  sourcecode(highlighted: highlighted, numbering: numbering)[
    #body
  ]
}

#let hm_red = "#fb5555"
#let hm_white = "#fefeff"

#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt,
)

#let styledtable(table) = {
  align(center,
    block(
      radius: 5pt,
      stroke: 2pt,
      clip: true,
      width: 100%,
      breakable: true,
      table
    )
  )
}


// This theme is inspired by https://github.com/matze/mtheme
// The polylux-port was performed by https://github.com/Enivex

// Consider using:
// #set text(font: "Fira Sans", weight: "light", size: 20pt)
// #show math.equation: set text(font: "Fira Math")
// #set strong(delta: 100)
// #set par(justify: true)



// #let hm-red = rgb("#23373b")
#let hm-red = rgb("#fb5555") // hm red
// #let accent-color-dark = rgb("#eb811b")
#let accent-color-dark = hm-red
#let accent-color-light = hm-red.lighten(40%)
// #let hm-white = white.darken(2%)
#let hm-white = rgb("#fefeff") //#hm white
#let text-color = rgb("#1a1c1f") // greyish black

#let m-footer = state("m-footer", [])

#let m-cell = block.with(
  width: 100%,
  height: 100%,
  above: 0pt,
  below: 0pt,
  breakable: false
)

#let m-progress-bar = utils.polylux-progress( ratio => {
  grid(
    columns: (ratio * 100%, 1fr),
    m-cell(fill: accent-color-dark),
    m-cell(fill: accent-color-light)
  )
})

#let metropolis-theme(
  aspect-ratio: "16-9",
  footer: [],
  body
) = {
  set page(
    paper: "presentation-" + aspect-ratio,
    fill: hm-white,
    margin: 0em,
    header: none,
    footer: none,
  )

  m-footer.update(footer)

  body
}

#let title-slide(
  title: [],
  subtitle: none,
  author: none,
  date: none,
  extra: none,
) = {
  let content = {
    set text(fill: text-color)
    set align(horizon)
    block(width: 100%, inset: 2em, {
      text(size: 1.3em, strong(title))
      if subtitle != none {
        linebreak()
        text(size: 0.9em, subtitle)
      }
      line(length: 100%, stroke: .05em + accent-color-dark)
      set text(size: .8em)
      if author != none {
        block(spacing: 1em, author)
      }
      if date != none {
        block(spacing: 1em, date)
      }
      set text(size: .8em)
      if extra != none {
        block(spacing: 1em, extra)
      }
    
    })
  }

  logic.polylux-slide(content)
}

#let slide(title: none, body) = {
  let header = {
    set align(top)
    if title != none {
      show: m-cell.with(fill: hm-red, inset: 1em)
      set align(horizon)
      set text(fill: hm-white, size: 1.2em)
      strong(title)
    } else { [] }
  }

  let footer = {
    set text(size: 0.8em)
    show: pad.with(.5em)
    set align(bottom)
    text(fill: hm-red.lighten(40%), m-footer.display())
    h(1fr)
    text(fill: hm-red, logic.logical-slide.display())
  }

  set page(
    header: header,
    footer: footer,
    margin: (top: 3em, bottom: 1em),
    fill: hm-white,
  )

  let content = {
    show: align.with(horizon)
    show: pad.with(2em)
    set text(fill: text-color)
    body
  }

  logic.polylux-slide(content)
}

#let new-section-slide(name) = {
  let content = {
    utils.register-section(name)
    set align(horizon)
    show: pad.with(20%)
    set text(size: 1.5em)
    name
    v(0.2em)
    block(height: 2pt, width: 100%, spacing: 0pt, m-progress-bar)
  }
  logic.polylux-slide(content)
}

#let focus-slide(body) = {
  set page(fill: hm-red, margin: 2em)
  set text(fill: hm-white, size: 1.5em)
  logic.polylux-slide(align(horizon + center, body))
}

#let alert = text.with(fill: accent-color-dark)

#let metropolis-outline = utils.polylux-outline(enum-args: (tight: false,))


