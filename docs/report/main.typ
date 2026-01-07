
#import "@preview/supercharged-hm:0.1.0": *
#import "glossary.typ" : glossary
#import "appendix.typ": appendix

#show: hm-template(
  title: "Traenslenzor",
  subtitle: [Advanced Deep Learning],
  authors: authors(
    "Jan Duchscherer",
    "Benedikt Köhler",
    "Jan Philip Schaible",
    "Felix Schladt",
    "Lukas Röß",
  ),
  toc-depth: 2,
  language: "en",
  version: none,
  // appendix: "/appendix.typ", # there is a bug in the template, I hope i will fix it soon, till then do it ugly...
  glossary: glossary, 
  bibliography: bibliography("sources.bib"),
)[
  // Include all chapters
  #include "/chapters/01_introduction.typ"
  #include "/chapters/02_components.typ"
  #include "/chapters/03_team.typ"

  // Ugly workaround till i will be able to the template
  #appendix
]
