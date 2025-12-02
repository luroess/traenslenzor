#import "template.typ": *

#show: slides.with(
  title: "Update Meeting - tränslenzor",
  date: datetime.today().display("[day]. [month repr:long] [year]"),
  ratio: 16/9,
  layout: "medium",
  title-color: rgb("#fb5555"),
)


== Supervisor
- Session System
  - Problem: LLM confused by FileIDs
  - Memory System: LLM can store Session-IDs (did not work)
  - Session-ID System: LLM gives Session-ID to Tools
- LLM Successive Tool Calls
  - Problem: LLM only calls one tool
  - Template adjustments: LLM now calls only Tools
- MCP Mocks for missing Tools
  - Unified MCP Server setup and startup procedure
  - more sophisticated mocks for missing tools

== Supervisors Problems
- LLM(llama3.2) is too stupid
  - Calls too many tools or not enough
  - Hallucinates a lot
  - Calls tools incorrectly / wrong order
  - Experiments with qwen3:8B are promising

== Text Extractor
- Working document deskewing (cv2)
- PaddleOCR log fixes
  - Paddle messes with python logging setup
    - PR Merged: https://github.com/PaddlePaddle/Paddle/pull/76699
- Session ID System integration
  