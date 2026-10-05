"""The local, per-user store of build-time snapshots. Never part of the repository, never shared with CI."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
KIND = "local-build-snapshot"
ENV_VAR = "PQ_BUILD_TIMES_DIR"


def data_dir(override=None, environ=None):
    """--data-dir, else $PQ_BUILD_TIMES_DIR, else $XDG_DATA_HOME/pq-build-times, else ~/.local/share/..."""
    environ = os.environ if environ is None else environ
    if override:
        return Path(override).expanduser()
    if environ.get(ENV_VAR):
        return Path(environ[ENV_VAR]).expanduser()
    base = environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "pq-build-times"


def snapshot_id(moment=None):
    return (moment or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_snapshot(root, snapshot):
    path = Path(root) / "snapshots" / f"{snapshot['id']}.json"
    _write_json(path, snapshot)
    return path


def load_snapshots(root, fingerprint=None):
    """All snapshots (oldest first), optionally of one fingerprint. Unreadable or foreign files are skipped."""
    snapshots = []
    for path in sorted((Path(root) / "snapshots").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"warning: skipping unreadable snapshot {path}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or data.get("kind") != KIND or data.get("schema_version") != SCHEMA_VERSION:
            print(f"warning: skipping {path} (not a version {SCHEMA_VERSION} snapshot)", file=sys.stderr)
            continue
        if fingerprint is None or data.get("fingerprint_id") == fingerprint:
            snapshots.append(data)
    return sorted(snapshots, key=lambda item: item["id"])


def load_baselines(root):
    path = Path(root) / "baselines.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def set_baseline(root, fingerprint, snapshot_ident):
    baselines = load_baselines(root)
    baselines[fingerprint] = snapshot_ident
    _write_json(Path(root) / "baselines.json", baselines)


def baseline_snapshot(root, fingerprint, snapshots=None):
    """The pinned baseline snapshot of a fingerprint, or None."""
    wanted = load_baselines(root).get(fingerprint)
    if wanted is None:
        return None
    for snapshot in snapshots if snapshots is not None else load_snapshots(root, fingerprint):
        if snapshot["id"] == wanted:
            return snapshot
    return None


def pinned_targets(root, fingerprint):
    """The touch targets a series is measured with: those of its baseline, else of its first snapshot."""
    snapshots = load_snapshots(root, fingerprint)
    reference = baseline_snapshot(root, fingerprint, snapshots) or (snapshots[0] if snapshots else None)
    return dict(reference.get("targets", {})) if reference else {}


def fingerprints(root):
    """{fingerprint id: [snapshots]} of everything in the store."""
    grouped = {}
    for snapshot in load_snapshots(root):
        grouped.setdefault(snapshot["fingerprint_id"], []).append(snapshot)
    return grouped
