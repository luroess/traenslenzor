# trÄnslenzor

## Prerequisites

- [uv](https://docs.astral.sh/uv/install/) - Fast Python package installer (required)

## Run Project

```sh
uv run python -m traenslenzor
```

## Setup (for Development)

After cloning, install python dependencies

```sh
uv sync
```

install pre-commit hooks:

```sh
uv run pre-commit install
```

## Running only the supervisor

```sh
uv run python -m traenslenzor.supervisor.supervisor
```

## Interface Definition
The individual components are called with a sessionId. This session id is used to retrieve a global shared state from the fileserver upon which is operated.

```mermaid
sequenceDiagram
    Note over Supervisor, FileServer: (0) Initialize Session
    Supervisor ->> FileServer: initializeSession
    FileServer ->> Supervisor: sessionId

    Note over Supervisor, DocumentLoader: (1) Load Document
    Supervisor ->> DocumentLoader: loadDocument(sessionId)
    DocumentLoader ->> FileServer: uploadDocument
    DocumentLoader ->> FileServer: setState(sessionId, state)

    Note over Supervisor, TextExtractor: (2) Extract Text
    Supervisor ->> TextExtractor: extractText(sessionId)
    TextExtractor ->> FileServer: getState(sessionId)
    TextExtractor ->> FileServer: getDocument(rawDocumentId)
    TextExtractor ->> FileServer: setState(sessionId, State)

    Note over Supervisor, FontDetector: (3) Detect Font
    Supervisor ->> FontDetector: detectFont(sessionId)
    FontDetector ->> FileServer: getState(sessionId)
    FontDetector ->> FileServer: getDocument(extractedDocumentId)
    FontDetector ->> FileServer: setState(sessionId, state)

    Note over Supervisor, DocumentClassDetector: (4) Classify Document
    Supervisor ->> DocumentClassDetector: classifyDocument(sessionId)
    DocumentClassDetector ->> FileServer: getState(sessionId)
    DocumentClassDetector ->> FileServer: getDocument(extractedDocumentId)
    DocumentClassDetector ->> FileServer: setState(sessionId, state)

    Note over Supervisor, Translator: (5) Translate Document
    Supervisor ->> Translator: translateText(sessionId)
    Translator ->> FileServer: getState(sessionId)
    Translator ->> FileServer: setState(sessionId, state)

    Note over Supervisor, Renderer: (6) Render Document
    Supervisor ->> Renderer: renderImage(sessionId)
    Renderer ->> FileServer: getState(sessionId)
    Renderer ->> FileServer: setState(sessionId, state)
```


### State Definition Template

```ts
interface State {
    rawDocumentId: string;
    deskewBackend?: "opencv" | "uvdoc";
    extractedDocument: {
        id: string,
        documentCoordinates: [];
        mapXYId?: string,
        mapXYShape?: [number, number, number],
        backend?: "opencv" | "uvdoc",
    }
    renderedDocumentId: string,
    text: TextItem[];
    language: string,

}

interface TextItem {
    extractedText: string;
    confidence: number;
    // 1st point upper left corner
    // 2nd point upper right corner
    // 3st point lower right corner
    // 4st point lower left corner
    bbox: {x: number, y: number}[];
    detectedFont: string;
    translatedText: string;
}
```
