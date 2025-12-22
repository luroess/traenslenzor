# Copilot Instructions for Traenslenzor

## Project Overview
Traenslenzor is a document translation system that processes images, detects layout and fonts, translates text, and re-renders the document. It uses a microservices architecture orchestrated by a Supervisor agent.

## Architecture
- **Supervisor (`traenslenzor/supervisor`)**: The central orchestrator built with LangGraph. It manages the workflow state and calls tools provided by MCP servers.
- **MCP Servers**: Independent components exposing tools via the Model Context Protocol (MCP) using `fastmcp`.
  - **Layout Detector (`traenslenzor/layout_detector`)**: Port 8001. Detects text and layout using PaddleOCR.
  - **Image Renderer (`traenslenzor/image_renderer`)**: Port 8002. Renders translated text onto images.
  - **Font Detector (`traenslenzor/font_detector`)**: Port 8003. Identifies fonts and estimates sizes.
- **File Server (`traenslenzor/file_server`)**: Port 8005. A FastAPI service for storing and retrieving intermediate files (images, JSON). Components use `FileClient` to interact with it.

## Key Technologies
- **Python 3.13+**
- **FastMCP**: For building MCP servers (`from fastmcp import FastMCP`).
- **LangGraph/LangChain**: For the supervisor agent logic.
- **FastAPI**: For the file server.
- **uv**: For dependency management (`uv sync`, `uv run`).

## Development Patterns
- **MCP Server Implementation**:
  - Use `FastMCP("Name")`.
  - Expose tools using `@mcp.tool`.
  - Run using `await mcp.run_async(transport="streamable-http", port=PORT, host=ADDRESS)`.
  - Define `ADDRESS` and `PORT` constants.
  - Expose `BASE_PATH` for the supervisor to connect.
- **Tool Integration**:
  - Register new MCP servers in `traenslenzor/supervisor/tools/mcp.py`.
- **File Handling**:
  - Do not pass large binary data directly through MCP.
  - Upload files to the File Server and pass the file ID/reference string.
  - Use `traenslenzor.file_server.client.FileClient` for upload/download.

## Common Commands
- **Install Dependencies**: `uv sync`
- **Run Tests**: `uv run pytest`
- **Run a Component**: `uv run python -m traenslenzor.<component>.server` (or specific entry point)

## Directory Structure
- `traenslenzor/`: Source code.
  - `supervisor/`: Agent logic.
  - `file_server/`: Storage service.
  - `font_detector/`, `layout_detector/`, `image_renderer/`: MCP servers.
- `tests/`: Tests.
- `docs/`: Documentation.
