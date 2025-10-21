#import "template.typ": *

#show: slides.with(
  title: "Update Meeting",
  date: datetime.today().display("[day]. [month repr:long] [year]"),
  ratio: 16/9,
  layout: "medium",
  title-color: blue.darken(60%),
)

== First Slide
