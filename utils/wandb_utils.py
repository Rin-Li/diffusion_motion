from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional


def _parse_tags(tags: Optional[str]) -> Optional[Iterable[str]]:
    if not tags:
        return None
    return [t.strip() for t in tags.split(",") if t.strip()]


def init_wandb(
    enabled: bool,
    *,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[str] = None,
    notes: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    group: Optional[str] = None,
):
    if not enabled:
        return None

    try:
        import wandb
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "W&B is enabled but 'wandb' is not installed. "
            "Install it or disable --wandb."
        ) from exc

    project = project or os.getenv("WANDB_PROJECT") or "diffusion_motion"
    entity = entity or os.getenv("WANDB_ENTITY")
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=_parse_tags(tags),
        notes=notes,
        group=group,
        config=config or {},
    )
    return run
