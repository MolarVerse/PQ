"""What the current branch changed against dev: the same counts as a merged PR, before it is one.

The diff is taken from the merge base of the base branch and the branch (what GitHub shows for a PR, "base...head"),
so changes that landed on dev in the meantime do not show up. Optionally the uncommitted work is included: changes
to tracked files (staged or not) and new untracked files. Areas, kinds and the counting rules are the ones of
cs_classify and cs_history, so the numbers match what `collect` will later report for the merged PR.
"""

from collections import namedtuple
from pathlib import Path

import cs_classify
import cs_history
import cs_output

Info = namedtuple("Info", "name base base_sha merge_base ahead behind uncommitted")

DEFAULT_BASES = ("origin/dev", "dev")


def resolve_base(repo, requested=None):
    for candidate in ([requested] if requested else DEFAULT_BASES):
        try:
            cs_history.git(repo, "rev-parse", "--verify", "--quiet", candidate + "^{commit}")
            return candidate
        except cs_history.HistoryError:
            continue
    raise cs_history.HistoryError(f"no usable base branch ({requested or ' or '.join(DEFAULT_BASES)}); pass --base")


def text_of(repo, *args):
    return cs_history.git(repo, *args).decode("utf-8", "replace").strip()


def count_untracked(repo):
    """{path: ("A", lines, 0)} of the files git does not track yet (and does not ignore)."""
    files = {}
    listing = cs_history.git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for item in listing:
        if not item:
            continue
        path = item.decode("utf-8", "replace")
        target = Path(repo) / path
        if target.is_symlink() or not target.is_file():
            continue
        content = target.read_bytes()
        binary = b"\0" in content[:8000]
        lines = 0 if binary else content.count(b"\n") + (1 if content and not content.endswith(b"\n") else 0)
        files[path] = ("A", lines, 0)
    return files


def read_branch(repo, base=None, ref="HEAD", uncommitted=False):
    """(Info, {path: (status, added, deleted)}) of the branch at ref against the merge base with the base branch."""
    if uncommitted and ref != "HEAD":
        raise cs_history.HistoryError("--uncommitted only works for the checked out branch (--ref HEAD)")
    base = resolve_base(repo, base)
    merge_base = text_of(repo, "merge-base", base, ref)
    name = text_of(repo, "rev-parse", "--abbrev-ref", ref)
    if name == "HEAD":
        name = "detached HEAD"
    ahead = int(text_of(repo, "rev-list", "--count", f"{merge_base}..{ref}"))
    behind = int(text_of(repo, "rev-list", "--count", f"{ref}..{base}"))
    diff = ["diff", "--raw", "--numstat", "--no-renames", merge_base] + ([] if uncommitted else [ref])
    files = cs_history.parse_files(cs_history.git(repo, *diff).decode("utf-8", "replace"))
    if uncommitted:
        files.update({path: value for path, value in count_untracked(repo).items() if path not in files})
    info = Info(name, base, text_of(repo, "rev-parse", "--short", base), text_of(repo, "rev-parse", "--short", merge_base),
                ahead, behind, uncommitted)
    return info, files


def sort_key(item):
    (area, kind), _ = item
    return (cs_classify.AREAS.index(area) if area in cs_classify.AREAS else len(cs_classify.AREAS), kind)


def table(rows, left=(0,)):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for index, row in enumerate(rows):
        out.append("  ".join(str(c).ljust(widths[i]) if i in left else str(c).rjust(widths[i]) for i, c in enumerate(row)).rstrip())
        if index == 0:
            out.append("  ".join("-" * w for w in widths))
    return out


def render(info, files, per_file=False):
    """The overview as lines of text."""
    splits = sorted(cs_history.split_of_change(files).items(), key=sort_key)
    head = f"{info.name} against {info.base} ({info.base_sha}): {info.ahead} commit{'s' if info.ahead != 1 else ''} ahead"
    head += f", {info.behind} behind (rebase or merge {info.base} before the PR)" if info.behind else ""
    out = [head]
    if info.uncommitted:
        out.append("including uncommitted changes (tracked files and new untracked files)")
    if not splits:
        return out + ["no counted changes"]

    rows = [["area / kind", "files", "A/D/M", "+lines", "-lines", "net"]]
    totals = [0, 0, 0, 0, 0, 0]
    for (area, kind), s in splits:
        numbers = [s.files_added + s.files_deleted + s.files_modified, s.files_added, s.files_deleted, s.files_modified, s.lines_added, s.lines_deleted]
        totals = [t + n for t, n in zip(totals, numbers)]
        rows.append([f"{area} / {kind}", numbers[0], f"{numbers[1]}/{numbers[2]}/{numbers[3]}", f"+{numbers[4]}", f"-{numbers[5]}", f"{numbers[4] - numbers[5]:+d}"])
    rows.append(["total", totals[0], f"{totals[1]}/{totals[2]}/{totals[3]}", f"+{totals[4]}", f"-{totals[5]}", f"{totals[4] - totals[5]:+d}"])
    out += [""] + table(rows)

    net = {}
    for (area, kind), s in splits:
        scope = cs_output.cpp_scope(area, kind)
        key = f"{scope[0]} {scope[1].replace('_', '-')}" if scope else cs_classify.group_of_area(area)
        net[key] = net.get(key, 0) + s.lines_added - s.lines_deleted
    summary = [f"{key} {value:+d}" for key, value in sorted(net.items())]
    out += ["", "net lines by scope (C++ only for prod/tests/perf): " + ", ".join(summary)]

    if per_file:
        counted = {path: value for path, value in files.items() if cs_classify.classify(path) is not None}

        def file_key(path):
            area, _, kind = cs_classify.classify(path)
            return sort_key(((area, kind), None)) + (path,)

        listing = [["", "+lines", "-lines", "file"]]
        for path in sorted(counted, key=file_key):
            status, added, deleted = counted[path]
            listing.append([status, f"+{added}", f"-{deleted}", path])
        out += [""] + table(listing, left=(0, 3))
    return out
