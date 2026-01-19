
#import "@preview/supercharged-hm:0.1.1": *
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
  glossary: glossary, 
  bibliography: bibliography("sources.bib"),
  appendix: appendix,
)[
  // Glossary formatting fix is pending: https://github.com/typst/packages/pull/3881

  // Include all chapters
  #include "/chapters/01_introduction.typ"
  #include "/chapters/02_components.typ"
  #include "/chapters/03_team.typ"
]
