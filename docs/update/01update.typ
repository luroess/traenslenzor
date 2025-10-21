#import "template.typ": *

#show: slides.with(
  title: "Update Meeting - tränslenzor",
  date: datetime.today().display("[day]. [month repr:long] [year]"),
  ratio: 16/9,
  layout: "medium",
  title-color: rgb("#fb5555"),
)


== General

- Task distribution in first group session
- Repository creation, Python setup
- Research and familiarization into the Topic


== Overview

Who does what?

#figure(caption: [Preliminary task distribution by components])[
  #image("imgs/whowhatwhenwhere.png")
]


== Supervisor
*What was done?*

- First experiments with langchain/langgraph
- Experimental langchain supervisor with dynamic tooling


*What questions are there?*

- Gemma3 seems to have no tooling support.
  - Is this known? There are hacks to work around this.
  - Can an alternative model like llama3.1 be used instead?

- Tools as Tools or as Nodes?
  - What are your experiences?

#pagebreak()

*Plan for the next two weeks*
- Connect first MCP server mocks
- Decision for a Langgraph design approach
- Document Provider
\
\
*TODOs*
- Couple counseling for Jan + Felix


== Plans for Next Week
*Jan S.:*
- void

*Felix:*
- void

*Jan D.:*
- void

*Benedikt:*
- void

*Lukas:*
- Read about MLP architecture
- Create concept for document translator
