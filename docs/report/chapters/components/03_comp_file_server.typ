#import "@preview/supercharged-hm:0.1.2": *

== File Server <comp_file_server>
The File Server component serves as the central node for sharing images, text, and additional metadata among the system's components.
It was introduced after alternative approaches for transferring auxiliary data via #gls("mcp") proved infeasible.
In particular, sharing image data through Base64 encoding was impractical, motivating the initial introduction of the File Server.
Consequently, an #gls("http") server based on FastAPI—referred to as the File Server—was introduced.

Later, additional session-related data was also migrated to the File Server.
This change was driven by the goal of minimizing the #gls("llm") tool parameter count: supplying numerous arguments to tool calls was found to significantly confuse the #gls("llm").

The system is based on a session mechanism identified by a unique `SessionId`, which allows every component to access and modify the shared `SessionState`.
When invoking a component, only the `SessionId` needs to be provided.

The File Server is the only backend component that maintains state, aside from the user frontend; all other components are stateless by design.

=== Session State
The session state, as shown in @session-state, represents the current state of a single session.
It is the core data structure shared across all components of the system.

#figure(caption: [File Server `SessionState` structure: `traenslenzor/file_server/session_state.py`])[
#code()[```py
class SessionState(BaseModel):
    rawDocumentId: str | None = None
    extractedDocument: ExtractedDocument | None = None
    superResolvedDocument: SuperResolvedDocument | None = None
    renderedDocumentId: str | None = None
    text: list[TextItem] | None = None
    language: str | None = None
    class_probabilities: dict[str, float] | None = None
```]]<session-state>

=== Session Progress
The `Session Progress` can be periodically fetched by the #gls("ui") to provide the user with progress information.
It is derived by the File Server through the information present in the `Session State` of a Session. The Supervisor does neither update nor accesses it.

#figure(caption: [File Server `ProgressState` and `SessionProgress` structures: `traenslenzor/file_server/session_state.py`])[
#code()[```py 
ProgressStage = Literal[
    "awaiting_document",
    "extracting_document",
    "detecting_language",
    "extracting_text",
    "translating",
    "detecting_font",
    "classifying",
    "rendering",
]

class SessionProgress(BaseModel):
    session_id: str
    stage: ProgressStage
    completed_steps: int
    total_steps: int
    steps: list[SessionProgressStep]
```]]
