from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path

from ..models import ValidationError


SKILL_NAME = "benchflow"
SKILL_ASSET_PATH = ("assets", "skills", SKILL_NAME)
SKILL_MARKER = ".benchflow-skill.json"


@dataclass(frozen=True, slots=True)
class SkillInstallResult:
    agent: str
    path: Path
    status: str


@dataclass(slots=True)
class _InstallTransaction:
    agent: str
    destination: Path
    transaction_root: Path
    staged: Path
    backup: Path
    had_existing: bool
    installed: bool = False


def _benchflow_version() -> str:
    try:
        return metadata.version("benchflow")
    except metadata.PackageNotFoundError:  # pragma: no cover - source-only fallback
        return "unknown"


def _source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _marker(destination: Path) -> dict[str, object] | None:
    if destination.is_symlink() or not destination.is_dir():
        return None
    marker_path = destination / SKILL_MARKER
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("managed_by") != "benchflow":
        return None
    return payload


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _destinations(
    *, home: Path | None = None, codex_home: Path | None = None
) -> list[tuple[str, Path]]:
    configured_home = os.environ.get("HOME", "").strip()
    resolved_home = (
        home or (Path(configured_home).expanduser() if configured_home else Path.home())
    ).resolve()
    configured_codex_home = os.environ.get("CODEX_HOME", "").strip()
    resolved_codex_home = (
        codex_home
        or (Path(configured_codex_home).expanduser() if configured_codex_home else None)
        or resolved_home / ".codex"
    ).resolve()
    return [
        ("Codex", resolved_codex_home / "skills" / SKILL_NAME),
        ("Claude", resolved_home / ".claude" / "skills" / SKILL_NAME),
    ]


def _stage_install(
    *,
    agent: str,
    destination: Path,
    source: Path,
    version: str,
    digest: str,
) -> _InstallTransaction:
    destination.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(
        tempfile.mkdtemp(prefix=".benchflow-skill-", dir=destination.parent)
    )
    staged = transaction_root / "new"
    try:
        shutil.copytree(source, staged)
        (staged / SKILL_MARKER).write_text(
            json.dumps(
                {
                    "managed_by": "benchflow",
                    "skill": SKILL_NAME,
                    "benchflow_version": version,
                    "content_sha256": digest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(transaction_root, ignore_errors=True)
        raise
    return _InstallTransaction(
        agent=agent,
        destination=destination,
        transaction_root=transaction_root,
        staged=staged,
        backup=transaction_root / "backup",
        had_existing=os.path.lexists(destination),
    )


def install_benchflow_skill(
    *,
    force: bool = False,
    home: Path | None = None,
    codex_home: Path | None = None,
    source_dir: Path | None = None,
    version: str | None = None,
) -> list[SkillInstallResult]:
    """Install the packaged BenchFlow skill for Codex and Claude Code."""
    destinations = _destinations(home=home, codex_home=codex_home)
    if len({destination for _, destination in destinations}) != len(destinations):
        raise ValidationError(
            "Codex and Claude skill destinations resolve to the same path; "
            "set CODEX_HOME to a distinct Codex configuration directory"
        )
    resolved_version = version or _benchflow_version()

    with ExitStack() as stack:
        source = source_dir
        if source is None:
            traversable = resources.files("benchflow")
            for part in SKILL_ASSET_PATH:
                traversable = traversable.joinpath(part)
            source = stack.enter_context(resources.as_file(traversable))
        source = Path(source)
        if not (source / "SKILL.md").is_file():
            raise ValidationError(
                f"packaged BenchFlow skill is missing SKILL.md: {source}"
            )
        digest = _source_digest(source)

        results: list[SkillInstallResult] = []
        pending: list[tuple[str, Path, str]] = []
        for agent, destination in destinations:
            exists = os.path.lexists(destination)
            marker = _marker(destination) if exists else None
            if exists and marker is None and not force:
                raise ValidationError(
                    f"refusing to overwrite unmanaged {agent} skill at {destination}; "
                    "rerun with --force to replace it"
                )
            if (
                marker
                and marker.get("content_sha256") == digest
                and marker.get("benchflow_version") == resolved_version
            ):
                results.append(SkillInstallResult(agent, destination, "unchanged"))
                continue
            pending.append((agent, destination, "updated" if exists else "installed"))

        transactions: list[_InstallTransaction] = []
        try:
            for agent, destination, _ in pending:
                transactions.append(
                    _stage_install(
                        agent=agent,
                        destination=destination,
                        source=source,
                        version=resolved_version,
                        digest=digest,
                    )
                )

            for transaction in transactions:
                if transaction.had_existing:
                    os.replace(transaction.destination, transaction.backup)
                os.replace(transaction.staged, transaction.destination)
                transaction.installed = True
        except Exception:
            for transaction in reversed(transactions):
                if transaction.installed and os.path.lexists(transaction.destination):
                    _remove_path(transaction.destination)
                if os.path.lexists(transaction.backup):
                    os.replace(transaction.backup, transaction.destination)
            for transaction in transactions:
                if transaction.transaction_root.exists():
                    shutil.rmtree(transaction.transaction_root)
            raise
        else:
            for transaction in transactions:
                if transaction.transaction_root.exists():
                    shutil.rmtree(transaction.transaction_root)

        result_by_agent = {result.agent: result for result in results}
        for agent, destination, status in pending:
            result_by_agent[agent] = SkillInstallResult(agent, destination, status)
        return [result_by_agent[agent] for agent, _ in destinations]
