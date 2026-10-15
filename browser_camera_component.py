from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_component = components.declare_component(
    "plate_browser_camera",
    path=str(Path(__file__).resolve().parent / "browser_camera"),
)


def browser_camera(
    *,
    key: str,
    camera_facing_mode: str = "environment",
    max_width: int = 640,
    capture_interval_ms: int = 1100,
) -> dict[str, Any] | None:
    return _component(
        key=key,
        cameraFacingMode=camera_facing_mode,
        maxWidth=max_width,
        captureIntervalMs=capture_interval_ms,
        default=None,
    )
