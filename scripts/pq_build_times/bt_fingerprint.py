"""Hardware and toolchain fingerprint of a local build-time snapshot.

Only snapshots with the same fingerprint id are compared or plotted together:
timings of a laptop and a workstation, or of two compilers, must never be mixed.
Everything that influences build time goes into the id; details that only help
reading the data (kernel, RAM, commit, tool versions) are recorded but do not.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess

ID_KEYS = ("arch", "cpu_model", "logical_cores", "compiler", "build_type", "target", "cmake_args", "jobs", "ccache")


def default_run(command, cwd=None):
    """Output of a command, or "" if it cannot be run."""
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def first_line(text):
    return text.strip().splitlines()[0].strip() if text.strip() else "unknown"


def cpu_model(run=default_run, cpuinfo="/proc/cpuinfo"):
    try:
        with open(cpuinfo, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                name, _, value = line.partition(":")
                if name.strip() in ("model name", "Model name") and value.strip():
                    return " ".join(value.split())
    except OSError:
        pass
    for command in (["lscpu"], ["sysctl", "-n", "machdep.cpu.brand_string"]):
        output = run(command)
        for line in output.splitlines():
            if command[0] == "lscpu" and line.startswith("Model name:"):
                return " ".join(line.split(":", 1)[1].split())
            if command[0] == "sysctl" and line.strip():
                return " ".join(line.split())
    return platform.processor() or "unknown"


def ram_gb(meminfo="/proc/meminfo"):
    try:
        with open(meminfo, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def os_name():
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return f"{platform.system()} {platform.release()}"


def comparable_cmake_args(args):
    """The CMake arguments that influence timings (machine-specific cache paths do not)."""
    return sorted(arg for arg in args if not arg.startswith("-DFETCHCONTENT_"))


def collect(compiler_path, build_type, target, cmake_args, jobs, ccache, run=default_run):
    """The fingerprint of this machine and configuration."""
    compiler_line = first_line(run([compiler_path, "--version"])) if compiler_path else "unknown"
    return {
        "arch": platform.machine() or "unknown",
        "cpu_model": cpu_model(run),
        "logical_cores": os.cpu_count() or 0,
        "ram_gb": ram_gb(),
        "os": os_name(),
        "kernel": platform.release(),
        "compiler": compiler_line,
        "linker": first_line(run(["ld", "--version"])) if shutil.which("ld") else "unknown",
        "cmake": first_line(run(["cmake", "--version"])),
        "ninja": first_line(run(["ninja", "--version"])),
        "build_type": build_type,
        "target": target,
        "cmake_args": comparable_cmake_args(cmake_args),
        "jobs": jobs,
        "ccache": ccache,
    }


def fingerprint_id(fingerprint):
    """Twelve hex characters over the keys that influence build time."""
    payload = json.dumps({key: fingerprint.get(key) for key in ID_KEYS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def describe(fingerprint):
    """A one-line description for tables and graph titles."""
    return (
        f"{fingerprint.get('arch')} | {fingerprint.get('cpu_model')} | {fingerprint.get('logical_cores')} threads | "
        f"{fingerprint.get('compiler')} | {fingerprint.get('build_type')} | target {fingerprint.get('target')} | "
        f"-j{fingerprint.get('jobs')}"
        + (" | ccache" if fingerprint.get("ccache") else "")
    )
