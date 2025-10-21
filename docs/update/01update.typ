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


== Roadmap

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, left),
  stroke: none,
  table.header(
    [*Milestone*], [*Duration*], [*Timeline*], [*Deliverables*]
  ),
  table.hline(),
  
  [*M1* – Module Prototyping & Architecture],
  [7 weeks],
  [Week 3–9],
  [
    - Prototype MCP servers
    - Working supervisor skeleton
    - First end-to-end pipeline mock
  ],
  
  table.hline(stroke: 0.5pt),
  
  [*M2* – Integration & Refinement],
  [2 weeks],
  [Week 10–11],
  [
    - Fully integrated pipeline
    - Core testing metrics
    - UI mock integration
  ],
  
  table.hline(stroke: 0.5pt),
  
  [*M3* – Evaluation & Optimization],
  [2 weeks],
  [Week 12–13],
  [
    - Evaluation report
    - Performance plots
    - Final system demo
  ],
  
  table.hline(stroke: 0.5pt),
  
  [*M4* – Documentation & Delivery],
  [2 weeks],
  [Week 14–15],
  [
    - Complete documentation
    - Demo video
    - Final submission
  ],
)


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

- Muss MCP wirklich für alle Tools verwendet werden?
  - Z.B. Image provider ist schwierig da wir nicht einfach Bilder über MCP teilen können (zumindest wenig sinnvoll)

#pagebreak()

*Plan for the next two weeks*
- Connect first MCP server mocks
- Decision for a Langgraph design approach
- Document Provider
\
\
*TODOs*
- Couple counseling for Jan + Felix


== Document Class Detector

*What has been done?*

- Setup ML Infrastructure:
  - PyTorch Lightning Module
    - `nn.CrossEntropyLoss`
    - `torchmetrics.Accuracy` & `torchmetrics.ConfusionMatrix`
    - setup logging, custom TQDM progress bar
  - Config Handling using Pydantic with _ConfigAsFactory_ design pattern.
  - Augmentation Pipeline using Albumentations
  - Optuna and WandB Integration for Hyperparameter Tuning and Experiment Tracking


== Plans for Next Week
*Jan D.:*
- Update and optimize factory for LitTrainer
  - Optimizer Config, LR Scheduler, Callbacks, ...
- Implement Dataset Class
- Implement AlexNet from scratch
- Improve config handling and logging & satisfy _ruff_

*Benedikt:*
- void

*Lukas:*
- Read about MLP architecture
- Create concept for document translator
