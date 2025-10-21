#import "template.typ": *

#show: slides.with(
  title: "Update Meeting - tränslenzor",
  date: datetime.today().display("[day]. [month repr:long] [year]"),
  ratio: 16/9,
  layout: "medium",
  title-color: rgb("#fb5555"),
)


== General

- Arbeitsaufteilung in erster Gruppensitzung
- Repository Erstellung, Python setup 
- Research and familiarization into the Topic


== Overview

Wer was wie wann wo.

#figure(caption: [Vorläufige Arbeitsaufteilung nach Komponenten])[
  #image("imgs/whowhatwhenwhere.png")
]

== Supervisor
\
*Was wurde gemacht?*

- Erste experimente mit langschain/langgraph
- Experimenteller langchain supervisor mit dynamic tooling


\
\
*Was für Fragen gibt es?*

- Gemma3 scheint keinen Tooling support zu haben. 
  - Ist das bekannt? Es gibt hacks um das zu umgehen. 
  - Kann alternativ auch ein Modell wie llama3.1 verwendet werden?

- Tools als Tools oder als Nodes?
  - Was sind ihre Erfahrungen?

#pagebreak()

\

*Plan für die nächsten zwei Wochen*
- Erste MCP server mocks anbinden
- Entscheidung für einen Langgraph Design Ansatz
- Document Provider
\
\
*TODOs*
- Couple counseling für Jan + Felix
