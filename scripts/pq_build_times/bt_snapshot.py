"""Take one build-time snapshot: a fixed set of scenarios, repeated, plus the exact include graph."""

import os
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone

import bt_fingerprint
import bt_ninja
import bt_store

SCENARIOS = ("cold", "noop", "touch_leaf", "touch_header_top", "touch_header_median")
TOUCH_TARGET = {"touch_leaf": "leaf", "touch_header_top": "header_top", "touch_header_median": "header_median"}
MARKER = ".pqbt-build-dir"
DEFAULT_ARGS = (
    "-G", "Ninja",
    "-DBUILD_WITH_NATIVE=OFF", "-DBUILD_WITH_ASE=OFF", "-DBUILD_WITH_DOCS=OFF", "-DBUILD_WITH_TESTS=ON",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=", "-DCMAKE_C_COMPILER_LAUNCHER=",
)


class SnapshotError(RuntimeError):
    pass


def run_command(command, cwd=None):
    """(return code, combined output)."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


class Builder:
    """Configures and builds in a directory this tool owns, and measures it."""

    def __init__(self, source_root, build_dir, deps_dir, build_type, cmake_args, target, jobs,
                 run=run_command, clock=time.perf_counter):
        self.source_root = os.path.realpath(source_root)
        self.build_dir = os.path.abspath(build_dir)
        self.deps_dir = os.path.abspath(deps_dir)
        self.build_type = build_type
        self.target = target
        self.jobs = jobs
        self.run = run
        self.clock = clock
        self.cmake_args = list(DEFAULT_ARGS) + [f"-DCMAKE_BUILD_TYPE={build_type}"] + list(cmake_args)
        self.fetch_args = [f"-DFETCHCONTENT_BASE_DIR={self.deps_dir}", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON"]

    # --- directory handling -------------------------------------------------
    def claim_build_dir(self):
        """Create the build directory, or accept one this tool created; never take over foreign content."""
        if os.path.isdir(self.build_dir):
            if os.path.exists(os.path.join(self.build_dir, MARKER)):
                return
            if os.listdir(self.build_dir):
                raise SnapshotError(
                    f"{self.build_dir} exists and was not created by this tool; use an empty or new --build-dir"
                )
        os.makedirs(self.build_dir, exist_ok=True)
        open(os.path.join(self.build_dir, MARKER), "w").close()

    def wipe_build_dir(self):
        if not os.path.exists(os.path.join(self.build_dir, MARKER)):
            raise SnapshotError(f"refusing to delete {self.build_dir}: not created by this tool")
        shutil.rmtree(self.build_dir)
        self.claim_build_dir()

    # --- steps --------------------------------------------------------------
    def configure(self):
        started = self.clock()
        code, output = self.run(["cmake", "-S", self.source_root, "-B", self.build_dir] + self.cmake_args + self.fetch_args)
        if code != 0:
            raise SnapshotError("configure failed:\n" + "\n".join(output.splitlines()[-25:]))
        return self.clock() - started

    def is_configured(self):
        return os.path.exists(os.path.join(self.build_dir, "build.ninja"))

    def _log_path(self):
        return os.path.join(self.build_dir, ".ninja_log")

    def _read_log(self):
        try:
            with open(self._log_path(), encoding="utf-8", errors="replace") as handle:
                return bt_ninja.log_lines(handle.read())
        except OSError:
            return set()

    def build(self):
        """One timed build of the target: wall seconds and what the build log says about the steps."""
        before = self._read_log()
        started = self.clock()
        code, output = self.run(["ninja", "-C", self.build_dir, "-j", str(self.jobs), self.target])
        wall = self.clock() - started
        if code != 0:
            raise SnapshotError("build failed:\n" + "\n".join(output.splitlines()[-25:]))
        after = self._read_log()
        result = bt_ninja.summarise_steps(bt_ninja.parse_log(after - before))
        result["wall_s"] = round(wall, 3)
        return result

    def cold(self):
        """Configure and build from nothing. The dependency cache is filled first so the network is not timed."""
        self.claim_build_dir()
        if not os.path.isdir(self.deps_dir):
            self.configure()
        self.wipe_build_dir()
        configure_s = self.configure()
        result = self.build()
        result["configure_s"] = round(configure_s, 3)
        return result

    def ensure_built(self):
        self.claim_build_dir()
        if not self.is_configured():
            raise SnapshotError("nothing to measure yet: include the 'cold' scenario in the first snapshot")
        self.build()

    def deps_text(self):
        code, output = self.run(["ninja", "-C", self.build_dir, "-t", "deps"])
        if code != 0:
            raise SnapshotError("ninja -t deps failed")
        return output

    def touch(self, relative):
        path = os.path.join(self.source_root, relative)
        if not os.path.exists(path):
            raise SnapshotError(f"cannot touch {relative}: no such file")
        os.utime(path)

    def compiler_path(self):
        try:
            with open(os.path.join(self.build_dir, "CMakeCache.txt"), encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("CMAKE_CXX_COMPILER:"):
                        return line.split("=", 1)[1].strip()
        except OSError:
            pass
        return ""


def aggregate(runs):
    """The median of repeated runs of one scenario, and how much they spread."""
    walls = [run["wall_s"] for run in runs]
    median = statistics.median(walls)
    cpu = statistics.median([run["cpu_s"] for run in runs])
    aggregated = {
        "wall_s": round(median, 3),
        "cpu_s": round(cpu, 3),
        "parallelism": round(cpu / median, 2) if median else None,
        "spread": round((max(walls) - min(walls)) / median, 3) if median else None,
        "steps": int(statistics.median([run["steps"] for run in runs])),
        "steps_stable": len({run["steps"] for run in runs}) == 1,
        "link_s": round(statistics.median([run["link_s"] for run in runs]), 3),
        "tail_s": round(statistics.median([run["tail_s"] for run in runs]), 3),
        "runs": runs,
    }
    if "configure_s" in runs[0]:
        aggregated["configure_s"] = round(statistics.median([run["configure_s"] for run in runs]), 3)
    return aggregated


def git_state(source_root, run=bt_fingerprint.default_run):
    def git(*args):
        return run(["git", "-C", source_root, *args]).strip()

    submodules = {}
    for line in git("submodule", "status").splitlines():
        parts = line.strip().lstrip("+-U").split()
        if len(parts) >= 2:
            submodules[parts[1]] = parts[0][:12]
    return {
        "commit": git("rev-parse", "HEAD")[:12] or "unknown",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "submodules": submodules,
    }


def take_snapshot(builder, store_root, scenarios, repeat, cold_runs, note, overrides, ccache,
                  now=lambda: datetime.now(timezone.utc), loadavg=os.getloadavg, log=print):
    """Run the scenarios and return the snapshot (not yet saved)."""
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        raise SnapshotError(f"unknown scenario(s): {', '.join(unknown)}; choose from {', '.join(SCENARIOS)}")
    started = now()
    load_start = loadavg()[0]
    results = {}

    if "cold" in scenarios:
        log(f"cold build x{cold_runs} (this is the long one)")
        results["cold"] = aggregate([builder.cold() for _ in range(cold_runs)])
    else:
        builder.ensure_built()

    graph = bt_ninja.include_graph(bt_ninja.parse_deps(builder.deps_text()), builder.source_root, builder.build_dir)
    fingerprint = bt_fingerprint.collect(
        builder.compiler_path(), builder.build_type, builder.target, builder.cmake_args, builder.jobs, ccache
    )
    fingerprint_id = bt_fingerprint.fingerprint_id(fingerprint)
    targets = bt_ninja.choose_targets(graph)
    targets.update({key: value for key, value in bt_store.pinned_targets(store_root, fingerprint_id).items() if value})
    targets.update({key: value for key, value in overrides.items() if value})

    if "noop" in scenarios:
        log(f"no-op build x{repeat}")
        results["noop"] = aggregate([builder.build() for _ in range(repeat)])

    for scenario, key in TOUCH_TARGET.items():
        if scenario not in scenarios:
            continue
        if not targets.get(key):
            log(f"skipping {scenario}: no suitable file found")
            continue
        runs = []
        log(f"{scenario} ({targets[key]}) x{repeat}")
        for _ in range(repeat):
            builder.touch(targets[key])
            runs.append(builder.build())
        results[scenario] = aggregate(runs)
        results[scenario]["file"] = targets[key]

    load_end = loadavg()[0]
    return {
        "schema_version": bt_store.SCHEMA_VERSION,
        "kind": bt_store.KIND,
        "id": bt_store.snapshot_id(started),
        "created_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note or "",
        "fingerprint_id": fingerprint_id,
        "fingerprint": fingerprint,
        "git": git_state(builder.source_root),
        "load": {"start": round(load_start, 2), "end": round(load_end, 2)},
        "target": builder.target,
        "repeat": repeat,
        "targets": targets,
        "scenarios": results,
        "include_graph": graph["metrics"],
    }
