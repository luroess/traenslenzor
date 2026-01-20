// Copyright 2024 Felix Schladt https://github.com/FelixSchladt

#import "../colors.typ": *

#let authors(..authors, styled: false) = {
  let by = authors
    .pos()
    .join(", ", last: " und ")

  if styled {    
  align(left,
    emph(
      text(fill: hm_red, [#by])
    )
  )
  } else {
    text([#by])
  }
}

#let authors_list(authors) = {
  let by = authors.join(", ", last: " und ")

  align(left,
    emph(
      text(fill: hm_red, [#by])
    )
  )
}
