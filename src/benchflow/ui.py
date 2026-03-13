from __future__ import annotations

import argparse
import logging
import shutil
import sys
from typing import Iterable

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich_argparse import RichHelpFormatter as _RichHelpFormatter

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for bootstrap environments
    Console = None
    RichHandler = None
    Panel = None
    Table = None
    _RichHelpFormatter = argparse.RawDescriptionHelpFormatter
    RICH_AVAILABLE = False


def _help_width() -> int:
    width = shutil.get_terminal_size((100, 24)).columns
    return max(88, min(width, 120))


class HelpFormatter(_RichHelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=28, width=_help_width())


_console = (
    Console(stderr=False, soft_wrap=True, highlight=False) if RICH_AVAILABLE else None
)
_error_console = (
    Console(stderr=True, soft_wrap=True, highlight=False) if RICH_AVAILABLE else None
)


def emit(message: str = "", *, stderr: bool = False, end: str = "\n") -> None:
    if RICH_AVAILABLE:
        console = _error_console if stderr else _console
        console.print(message, end=end, markup=False, highlight=False)
        return
    stream = sys.stderr if stderr else sys.stdout
    print(message, file=stream, end=end)


def rule(title: str) -> None:
    if RICH_AVAILABLE:
        _console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        emit(f"== {title} ==")


def panel(title: str, rows: Iterable[tuple[str, str]]) -> None:
    items = [(label, value) for label, value in rows if value]
    if not items:
        return
    if RICH_AVAILABLE:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold white")
        table.add_column(style="white")
        for label, value in items:
            table.add_row(label, value)
        _console.print(Panel.fit(table, title=title, border_style="cyan"))
        return
    emit(title)
    for label, value in items:
        emit(f"  {label}: {value}")


def step(message: str) -> None:
    if RICH_AVAILABLE:
        _console.print(f"[bold cyan]›[/bold cyan] {message}")
    else:
        emit(f"> {message}")


def detail(message: str) -> None:
    if RICH_AVAILABLE:
        _console.print(f"  [dim]{message}[/dim]")
    else:
        emit(f"  {message}")


def success(message: str) -> None:
    if RICH_AVAILABLE:
        _console.print(f"[bold green]✓[/bold green] {message}")
    else:
        emit(f"OK {message}")


def warning(message: str) -> None:
    if RICH_AVAILABLE:
        _error_console.print(f"[bold yellow]![/bold yellow] {message}")
    else:
        emit(f"WARNING: {message}", stderr=True)


def error(message: str) -> None:
    if RICH_AVAILABLE:
        _error_console.print(f"[bold red]error[/bold red] {message}")
    else:
        emit(f"ERROR: {message}", stderr=True)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_benchflow_configured", False):
        root.setLevel(level.upper())
        return

    root.handlers.clear()
    if RICH_AVAILABLE:
        handler = RichHandler(
            show_time=False,
            show_level=True,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
        logging.basicConfig(
            level=level.upper(), format="%(message)s", handlers=[handler]
        )
    else:
        logging.basicConfig(level=level.upper(), format="%(levelname)s: %(message)s")
    root._benchflow_configured = True  # type: ignore[attr-defined]
