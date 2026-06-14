#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def fail(message, stdout=None):
    print(message, file=sys.stderr)
    if stdout:
        print("\n--- PQ output ---", file=sys.stderr)
        print(stdout, file=sys.stderr)
    return 1


def contains_all(text, needles):
    return [needle for needle in needles if needle not in text]


def contains_any(text, needles):
    return [needle for needle in needles if needle in text]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", required=True)
    parser.add_argument("--case-zip", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    pq = Path(args.pq)
    case_zip = Path(args.case_zip)

    if not pq.exists():
        return fail(f"PQ executable does not exist: {pq}")
    if not case_zip.exists():
        return fail(f"regression case zip does not exist: {case_zip}")

    with tempfile.TemporaryDirectory(prefix="pq-regression-") as tmp:
        case_dir = Path(tmp)
        with zipfile.ZipFile(case_zip) as archive:
            archive.extractall(case_dir)

        manifest_path = case_dir / args.manifest
        if not manifest_path.exists():
            return fail(f"manifest does not exist in case zip: {args.manifest}")

        manifest = json.loads(manifest_path.read_text())
        input_file = manifest["input"]
        timeout = manifest.get("timeout_seconds", 30)

        completed = subprocess.run(
            [str(pq), input_file],
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )

        stdout = completed.stdout
        if completed.returncode != 0:
            return fail(
                f"PQ exited with status {completed.returncode}",
                stdout,
            )

        missing_stdout = contains_all(
            stdout,
            manifest.get("stdout_contains", []),
        )
        if missing_stdout:
            return fail(
                f"PQ output is missing expected text: {missing_stdout}",
                stdout,
            )

        rejected_stdout = contains_any(
            stdout,
            manifest.get("stdout_rejects", []),
        )
        if rejected_stdout:
            return fail(
                f"PQ output contains rejected text: {rejected_stdout}",
                stdout,
            )

        for filename in manifest.get("expected_files", []):
            output = case_dir / filename
            if not output.exists():
                return fail(f"expected output file was not created: {filename}", stdout)
            if output.stat().st_size == 0:
                return fail(f"expected output file is empty: {filename}", stdout)

        for filename, needles in manifest.get("file_contains", {}).items():
            text = (case_dir / filename).read_text()
            missing = contains_all(text, needles)
            if missing:
                return fail(f"{filename} is missing expected text: {missing}", stdout)

        for filename, needles in manifest.get("file_rejects", {}).items():
            text = (case_dir / filename).read_text().lower()
            rejected = contains_any(text, [needle.lower() for needle in needles])
            if rejected:
                return fail(f"{filename} contains rejected text: {rejected}", stdout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
