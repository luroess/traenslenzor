"""CLI for running the doc-scanner on a local file using DocScannerMCPConfig."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from traenslenzor.doc_classifier.utils import Console
from traenslenzor.doc_scanner.backtransform import (
    backtransform_with_corners,
    backtransform_with_map_xy,
)
from traenslenzor.doc_scanner.configs import DocScannerMCPConfig
from traenslenzor.doc_scanner.runtime import DocScannerRuntime
from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import SessionState, initialize_session

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "doc-scanner.toml"


class CLIDocScannerConfig(DocScannerMCPConfig):
    """CLI-enabled config for deskewing a single file."""

    file_path: Path = Field(..., description="Path to the image file to deskew.")
    output_path: Path | None = Field(
        default=None,
        description="Optional output path for the deskewed image (PNG).",
    )
    map_xy_path: Path | None = Field(
        default=None,
        description="Optional output path for the map_xy .npy file (if generated).",
    )
    backtransform_path: Path | None = Field(
        default=None,
        description="Optional output path for the backtransformed image (PNG).",
    )
    backtransform_mask_path: Path | None = Field(
        default=None,
        description="Optional output path for the backtransform mask (PNG).",
    )
    backtransform_composite_path: Path | None = Field(
        default=None,
        description="Optional output path for the backtransform composite image (PNG).",
    )
    config_path: Path | None = Field(
        default=None,
        description="Optional path to a doc-scanner TOML config file.",
    )

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        protected_namespaces=(),
        cli_parse_args=True,
        env_prefix="DOC_SCANNER_",
        toml_file=Path(__file__).resolve().parents[2] / "config" / "doc-scanner.toml",
    )


def _resolve_output_path(file_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return file_path.with_name(f"{file_path.stem}_deskewed.png")


def _resolve_backtransform_paths(
    file_path: Path,
    output_path: Path,
    backtransform_path: Path | None,
    backtransform_mask_path: Path | None,
) -> tuple[Path, Path]:
    if backtransform_path is not None and backtransform_mask_path is not None:
        return backtransform_path, backtransform_mask_path

    base = output_path if output_path is not None else file_path
    default_back = base.with_name(f"{base.stem}_backtransformed.png")
    default_mask = base.with_name(f"{base.stem}_backtransform_mask.png")
    return backtransform_path or default_back, backtransform_mask_path or default_mask


def _resolve_backtransform_composite_path(
    file_path: Path,
    output_path: Path,
    backtransform_composite_path: Path | None,
) -> Path:
    if backtransform_composite_path is not None:
        return backtransform_composite_path
    base = output_path if output_path is not None else file_path
    return base.with_name(f"{base.stem}_backtransform_composite.png")


def _load_base_config(cli_config: CLIDocScannerConfig, console: Console) -> DocScannerMCPConfig:
    config_path = cli_config.config_path
    if config_path is None and _DEFAULT_CONFIG_PATH.exists():
        config_path = _DEFAULT_CONFIG_PATH

    if config_path is None:
        return DocScannerMCPConfig()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.stat().st_size == 0:
        return DocScannerMCPConfig()

    try:
        return DocScannerMCPConfig.from_toml(config_path)
    except Exception as exc:
        console.error(f"Failed to load doc-scanner config at {config_path}: {exc}")
        return DocScannerMCPConfig()


def _build_runtime_config(cli_config: CLIDocScannerConfig, console: Console) -> DocScannerMCPConfig:
    base_config = _load_base_config(cli_config, console)
    overrides = cli_config.model_dump(
        exclude={
            "file_path",
            "output_path",
            "map_xy_path",
            "backtransform_path",
            "backtransform_mask_path",
            "backtransform_composite_path",
            "config_path",
        },
        exclude_unset=True,
    )
    if not overrides:
        return base_config

    def _deep_merge(base: dict, updates: dict) -> dict:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = _deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    merged = _deep_merge(base_config.model_dump(), overrides)
    return DocScannerMCPConfig.model_validate(merged)


async def _run_cli(cli_config: CLIDocScannerConfig) -> None:
    console = Console.with_prefix("doc-scanner-cli", "run")
    config = _build_runtime_config(cli_config, console)
    console.set_verbose(config.verbose).set_debug(config.is_debug)

    file_path = cli_config.file_path.expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    DocScannerMCPConfig.model_rebuild(_types_namespace={"DocScannerRuntime": DocScannerRuntime})
    runtime = config.setup_target()

    input_img = Image.open(file_path)
    file_id = await FileClient.put_img(file_path.name, input_img)
    if file_id is None:
        raise RuntimeError(f"Failed to upload {file_path} to file server.")

    session = initialize_session()
    session.rawDocumentId = file_id

    session_id = await SessionClient.create(session)
    extracted = await runtime.scan_session(session_id)

    def update_session(state: SessionState) -> None:
        state.extractedDocument = extracted

    await SessionClient.update(session_id, update_session)

    output_path = _resolve_output_path(file_path, cli_config.output_path)
    output_img = await FileClient.get_image(extracted.id)
    if output_img is None:
        raise RuntimeError(f"Failed to download output image for id {extracted.id}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(output_path, format="PNG")
    console.log(f"Deskewed image saved to {output_path}")

    map_xy = None
    if extracted.mapXYId is not None:
        map_xy = await FileClient.get_numpy_array(extracted.mapXYId)
        if map_xy is None:
            console.warn(f"Failed to download map_xy for id {extracted.mapXYId}.")

    if cli_config.map_xy_path is not None:
        if extracted.mapXYId is None:
            console.warn("No map_xy generated for this run.")
        elif map_xy is None:
            console.warn(f"Failed to download map_xy for id {extracted.mapXYId}.")
        else:
            cli_config.map_xy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cli_config.map_xy_path, map_xy)
            console.log(f"map_xy saved to {cli_config.map_xy_path}")

    extracted_rgb = np.array(output_img.convert("RGB"), dtype=np.uint8)
    output_shape = (input_img.height, input_img.width)

    backtransform_img = None
    backtransform_mask = None
    if map_xy is not None:
        backtransform_img, backtransform_mask = backtransform_with_map_xy(
            extracted_rgb, map_xy, output_shape
        )
    elif extracted.documentCoordinates:
        backtransform_img, backtransform_mask = backtransform_with_corners(
            extracted_rgb, extracted.documentCoordinates, output_shape
        )
    else:
        console.warn("No map_xy or documentCoordinates available for backtransform.")

    if backtransform_img is not None and backtransform_mask is not None:
        back_path, mask_path = _resolve_backtransform_paths(
            file_path, output_path, cli_config.backtransform_path, cli_config.backtransform_mask_path
        )
        back_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(backtransform_img).save(back_path, format="PNG")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((backtransform_mask.astype(np.uint8) * 255)).save(
            mask_path, format="PNG"
        )
        console.log(f"Backtransformed image saved to {back_path}")
        console.log(f"Backtransform mask saved to {mask_path}")

        session = await SessionClient.get(session_id)
        raw_img = (
            await FileClient.get_image(session.rawDocumentId)
            if session.rawDocumentId
            else None
        )
        if raw_img is None:
            console.warn("Failed to fetch raw image for composite backtransform.")
        else:
            raw_rgb = np.array(raw_img.convert("RGB"), dtype=np.uint8)
            composite = raw_rgb.copy()
            composite[backtransform_mask] = backtransform_img[backtransform_mask]
            composite_path = _resolve_backtransform_composite_path(
                file_path, output_path, cli_config.backtransform_composite_path
            )
            composite_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(composite).save(composite_path, format="PNG")
            console.log(f"Backtransform composite saved to {composite_path}")

    console.plog(extracted.model_dump())


def main() -> None:
    cli_config = CLIDocScannerConfig()
    try:
        asyncio.run(_run_cli(cli_config))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        Console.with_prefix("doc-scanner-cli", "error").error(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
