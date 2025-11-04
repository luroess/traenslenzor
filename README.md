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
