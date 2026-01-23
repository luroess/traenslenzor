#import "@preview/supercharged-hm:0.1.2": *

== File Server <comp_file_server>

The file server component serves as the central node sharing image, text and additional metadata between the different components.
It was introduced due as other methods of sharing additional data via #gls("mcp") proved to be not viable.
Sharing image data via Base64 encoding is not really convenient, warranting the original introduction of the File Server. 
Later, additional metadata was moved into it, due to us trying to limit the #gls("llm") parameter size as much as possible, requiring multiple different arguments to the tool calls proved to confuse the #gls("llm") quite a lot.
Therefore an #gls("http") server based upon FastAPI dubbed "File Server" was introduced.\
Based upon a session system with a unique `SessionId`, every component can access and alter the `SessionState`. 
If a component is called, only the `SessionId` must be supplied.
This architecture allows for multiple #gls("ui") Clients utilizing the same backend runtime simultaneously.
The file server is the only instance keeping state except the user frontend. All other components are stateless in nature.

=== Session State
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
```]]

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
