// Copyright 2024 Felix Schladt https://github.com/FelixSchladt

#import "/template/colors.typ": *

#import "/template/libs/tablestyle.typ": *
#import "/template/libs/requirements.typ": *


#let risiko(
  Zielobjekt,
  Schutzbedarf,
  Gefährdung,
  Eintrittshäufigkeit,
  Auswirkung,
  Risiko,
  Beschreibung,
  Bewertung,
  Gegenmaßnahmen,
) = { 
  show table.header: set text(size: 15pt)
  show figure: set block(breakable: true)
  set block(breakable: true)
  figure(
    caption: Zielobjekt + " " + Gefährdung,
    supplement: "Risiko",
    kind: "Risiko",
    numbering: "1",
    box(
      stroke: 2pt,
      radius: 5pt,
      clip: true,
      table(
        columns: (23%, 77%),
        align: left,
        fill: (x, y) => {
          if (y == 0) {
            hm_grey_very_light
          }
        },
        table.header(
          [*Zielobjekt*], [*#Zielobjekt*]
        ),
        [*Schutzbedarf*],[#Schutzbedarf],
        [*Gefährdung*], [#Gefährdung],
        [*Eintrittshäufigkeit*], [#Eintrittshäufigkeit],
        [*Auswirkung*], [#Auswirkung],
        [*Risiko*], [#Risiko],
        [*Beschreibung*], [#Beschreibung],
        [*Bewertung*], [#Bewertung],
        [*Gegenmaßnahmen*], [#Gegenmaßnahmen]
      )
    )
  )
}