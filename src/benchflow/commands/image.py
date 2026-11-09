from __future__ import annotations

from pathlib import Path

import click

from ..toolbox.image import build_lmcache_image
from ..ui import success


@click.group(
    "image", help="Prepare serving images locally, outside benchmark execution."
)
def image_group() -> None:
    pass


@image_group.command(
    "build",
    help="Build, check, and publish a vLLM+LMCache image to Quay using Podman.",
)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the recipe and commands without calling Podman.",
)
def image_build(config: Path, dry_run: bool) -> None:
    result = build_lmcache_image(config, dry_run=dry_run)
    if result:
        success(f"Published {result['image']}")
        success(f"Image record: {config.resolve().with_suffix('.image.json')}")
