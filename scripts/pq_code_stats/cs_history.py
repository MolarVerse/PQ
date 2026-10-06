"""The first-parent history of a branch as changes (merged PRs, other merges, direct commits).

One `git log` call yields the diff of every change against its first parent, which for a merged PR is exactly
what the PR changed on the branch. The totals (files and lines per area and kind) are rolled forward change by
change from the root commit, and can be verified against a full count of the final tree.
"""

import re
import subprocess
from collections import Counter, namedtuple
from datetime import datetime, timezone

import cs_classify

Change = namedtuple("Change", "sha parents date author subject kind pr title branch")
Split = namedtuple("Split", "files_added files_deleted files_modified lines_added lines_deleted")
Record = namedtuple("Record", "change splits state")   # state: {(area, kind): (files, lines)} after the change

FORMAT = "%x01%H%x00%P%x00%cI%x00%an%x00%s%x00%b%x02"
MERGE_PR = re.compile(r"^Merge pull request #(\d+)\b(?: from [^/\s]+/(\S+))?")
SQUASH_PR = re.compile(r"\(#(\d+)\)\s*$")
NUMSTAT = re.compile(r"^(\d+|-)\t(\d+|-)\t(.+)$")
EMPTY_SPLIT = Split(0, 0, 0, 0, 0)
GITLINK = "160000"


class HistoryError(RuntimeError):
    pass


def git(repo, *args, data=None):
    result = subprocess.run(["git", "-c", "core.quotepath=false", "-C", str(repo), *args], input=data, capture_output=True)
    if result.returncode != 0:
        raise HistoryError(f"git {' '.join(args[:3])} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


def resolve_ref(repo, requested=None):
    for candidate in ([requested] if requested else ["origin/dev", "dev", "HEAD"]):
        try:
            git(repo, "rev-parse", "--verify", "--quiet", candidate + "^{commit}")
            return candidate
        except HistoryError:
            continue
    raise HistoryError(f"no usable ref ({requested or 'origin/dev, dev, HEAD'})")


def kind_and_title(subject, parents, body):
    merged = MERGE_PR.match(subject)
    if merged:
        title = next((line.strip() for line in body.splitlines() if line.strip()), subject)
        return "pr", int(merged.group(1)), title, merged.group(2)
    squashed = SQUASH_PR.search(subject)
    if squashed:
        return "pr", int(squashed.group(1)), SQUASH_PR.sub("", subject).strip(), None
    return ("merge" if len(parents) > 1 else "commit"), None, subject, None


def parse_log(text):
    """[(Change, {path: (status, added, deleted)})] in the order of the log (newest first)."""
    entries = []
    for block in text.split("\x01")[1:]:
        header, _, rest = block.partition("\x02")
        fields = header.split("\x00")
        if len(fields) != 6:
            raise HistoryError(f"unexpected log header: {header[:80]!r}")
        sha, parents, date, author, subject, body = fields
        parent_list = parents.split()
        kind, pr, title, branch = kind_and_title(subject, parent_list, body)
        moment = datetime.fromisoformat(date).astimezone(timezone.utc)
        files = {}
        for line in rest.splitlines():
            if line.startswith(":"):
                meta, _, path = line.partition("\t")
                if GITLINK in meta.split()[:2]:   # a submodule is a pointer, not code (the same rule as in count_tree)
                    continue
                files[path] = [meta.split()[-1][0], 0, 0]
            else:
                numbers = NUMSTAT.match(line)
                if numbers and numbers.group(3) in files:
                    added, deleted, path = numbers.groups()
                    files[path][1:] = [0 if added == "-" else int(added), 0 if deleted == "-" else int(deleted)]
        entries.append((Change(sha, parent_list, moment, author, subject, kind, pr, title, branch), {k: tuple(v) for k, v in files.items()}))
    return entries


def split_of_change(files):
    """{(area, kind): Split} of one change."""
    totals = {}
    for path, (status, added, deleted) in files.items():
        where = cs_classify.classify(path)
        if where is None:
            continue
        key = (where[0], where[2])
        old = totals.get(key, EMPTY_SPLIT)
        totals[key] = Split(
            old.files_added + (status == "A"), old.files_deleted + (status == "D"),
            old.files_modified + (status not in "AD"), old.lines_added + added, old.lines_deleted + deleted,
        )
    return totals


def roll_forward(entries):
    """Records oldest first, each with the state (files, lines per (area, kind)) after the change."""
    files, lines = Counter(), Counter()
    records = []
    for change, changed in reversed(entries):
        splits = split_of_change(changed)
        for key, split in splits.items():
            files[key] += split.files_added - split.files_deleted
            lines[key] += split.lines_added - split.lines_deleted
        state = {key: (files[key], lines[key]) for key in set(files) | set(lines) if files[key] or lines[key]}
        records.append(Record(change, splits, state))
    return records


def read_history(repo, ref):
    text = git(repo, "log", "--first-parent", "-m", "--root", "--raw", "--numstat", "--no-renames",
               f"--format={FORMAT}", ref).decode("utf-8", "replace")
    return roll_forward(parse_log(text))


def count_tree(repo, ref):
    """{(area, kind): (files, lines)} of the tree at ref, counted from the file contents (for verification)."""
    listing = git(repo, "ls-tree", "-r", "-z", ref).split(b"\0")
    entries = []
    for item in listing:
        if not item:
            continue
        meta, _, path = item.partition(b"\t")
        mode, kind, sha = meta.split()
        where = cs_classify.classify(path.decode("utf-8", "replace"))
        if kind == b"blob" and where is not None:
            entries.append((sha, (where[0], where[2])))
    blobs = git(repo, "cat-file", "--batch", data=b"".join(sha + b"\n" for sha, _ in entries))
    files, lines = Counter(), Counter()
    position = 0
    for sha, key in entries:
        end = blobs.index(b"\n", position)
        size = int(blobs[position:end].split()[2])
        content = blobs[end + 1:end + 1 + size]
        position = end + 1 + size + 1
        files[key] += 1
        if b"\0" not in content[:8000]:
            lines[key] += content.count(b"\n") + (1 if content and not content.endswith(b"\n") else 0)
    return {key: (files[key], lines[key]) for key in files}


def verify(records, counted):
    """Differences between the rolled-forward state and the counted tree: [(key, rolled, counted)]."""
    rolled = records[-1].state if records else {}
    keys = set(rolled) | set(counted)
    return sorted((key, rolled.get(key, (0, 0)), counted.get(key, (0, 0))) for key in keys
                  if rolled.get(key, (0, 0)) != counted.get(key, (0, 0)))
