#import "@preview/supercharged-hm:0.1.2": *
#import "@preview/wrap-it:0.1.1": wrap-content
#pagebreak()

== User Interface (Streamlit) <comp_user_interface>
*Jan Duchscherer*

The user interacts with trÄnslenzor through a web-based, chat-interface #gls("ui") implemented in Streamlit.
The app caches local state in `st.session_state` (chat history, caches, and the current `session_id`), while the File Server (@comp_file_server) acts as the persistent source of truth for all workflow artifacts and metadata (`SessionState` + files).
To keep the UI responsive, long-running work (supervisor run and #gls("http") calls) is executed asynchronously in a background event loop.
Implementation notes:

- The background async runner keeps a dedicated event loop and stores the pending `Future` in `st.session_state` for lifecycle tracking.
- Sidebar polling is enabled only while the supervisor is running (interval: 10s); otherwise the UI serves cached state.
- The image panel uses file-id-indexed caches for images and map_xy grids; the “Overlay” tab draws document corners and (optionally) the map_xy grid.
- Uploading a new document resets the session via `prepare_new_doc()` and creates a new File Server session.
- The UI exposes session import/export (pickle) and targeted deletion of session components.

#figure(
  caption: [Streamlit UI integration with File Server sessions and files.],
)[
  #image("/graphics/streamlit-app-sequence.svg")
] <fig-streamlit-app-sequence>

Control flow in @fig-streamlit-app-sequence:

- *Upload:* The UI uploads image bytes and stores the returned id in `SessionState.rawDocumentId`.
- *Run:* The supervisor executes tool chains that update `SessionState` and upload derived artifacts (deskewed, rendered, etc.).
- *Poll & render:* The UI polls `get_progress()` and `get()` to keep the sidebar session overview current, and downloads the referenced image files to keep the image panel (raw/extracted/rendered/super-res) up-to-date.

// #wrap-content(
//   align: top + right,
//   column-gutter: 30pt,
//   columns: (1fr, 1fr),
//   [#figure(
//     caption: [UI example: triggering super-resolution and document classification.],
//   )[
//     #image("/imgs/chat_super-res-classify.png", width: 100%)
//   ] <fig-ui-superres-classify>],
//   [
//     The sidebar exposes session-aware actions that trigger our scanner and classifier tools:

//     - *Super-res:* calls `super_resolve_document(session_id)` in the Doc Scanner (@comp_doc_scanner) and stores the resulting `SuperResolvedDocument` in the session.
//     - *Classify document:* calls `classify_document(session_id)` in the Document Class Detector (@comp_doc_cls) and displays the top result from `SessionState.class_probabilities`.

//     The image panel supports stage tabs (raw, extracted, rendered, super-res) to inspect intermediate artifacts.
//   ],
// )
