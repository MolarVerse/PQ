# AGENTS.md — Contributor Guide for PQ

This file documents the codebase structure, development workflows, and conventions that contributors and coding assistants must follow when contributing to PQ.

---

## Project Overview

PQ is an open-source **C++23** molecular dynamics simulation engine (classical MM, QM, and QM-RPMD) built with **CMake**, with optional **Python bindings** and **MPI**. It integrates with external QM engines (DFTB+, Turbomole, PySCF, MACE, FeNNol) via ASE. Licensed under GPLv3.

---

## Git Workflow — CRITICAL RULES

> **Contributors and coding assistants must follow these rules without exception.**

The project uses the **git-flow** branching model: `main` is the stable/release branch, `dev` is the integration branch.

1. **Never push directly to `main` or `dev`.** These are protected branches.
2. **Always create a feature/bugfix branch from `dev`** for any change (git-flow: `git flow feature start <name>`, or plain `git checkout -b feature/<name> dev`).
3. **Open pull requests targeting `dev`**, not `main`, unless explicitly instructed otherwise. Only release PRs go from `dev` to `main`, and those must contain `Release-x.y.z` in the title or body (enforced by CI).
4. Branch naming convention: `feature/<short-description>` or `bugfix/<short-description>`.
5. **Always `git checkout dev` and `git pull` to update local `dev` before creating any new branch.**
6. **Never amend or force-push** to shared branches.
7. Use `git push -u origin <branch-name>` when pushing a new branch.
8. **Install the commit-msg hook once per clone**: `cp .githooks/commit-msg .git/hooks/`. It rejects commits whose subject doesn't start with a recognized conventional-commit-style prefix (see below) — this is enforced locally, not just in CI.

```bash
# Correct workflow
git checkout dev
git pull
git checkout -b feature/my-feature
# ... make changes, commit ...
git push -u origin feature/my-feature
# Then open a PR targeting dev
```

---

## Commit Message Convention

Enforced by `.githooks/commit-msg` and mirrored in `cliff.toml` (used to generate `DEV-CHANGELOG.md` at release time). Every commit subject must start with one of these prefixes (optionally with a `(scope)`), followed by `: `:

| Group          | Prefixes                              |
| -------------- | -------------------------------------- |
| Feature        | `feat:`, `feature:`                    |
| Fix            | `fix:`, `bugfix:`                      |
| Docs           | `docs:`, `doc:`, `documentation:`      |
| Style          | `style:`, `format:`                    |
| Cleanup        | `cleanup:`, `clean-up:`, `clean:`      |
| Refactor       | `refactor:`, `ref:`                    |
| Performance    | `perf:`, `performance:`                |
| Test           | `test:`, `tests:`, `testing:`          |
| Internal       | `internal:`                            |
| Chore          | `chore:`                               |
| Build          | `build:`, `cmake:`, `deps:`            |
| CI / Workflow  | `ci:`, `workflow:`, `flow:`            |
| Breaking       | `breaking:`                            |
| Example        | `example:`, `examples:`                |
| Merge / Revert | `merge:`, `revert:`                    |
| Administrative | `admin:`, `administrative:`            |

Example: `docs: add shared contributor guide`

---

## Repository Structure

```
PQ/
├── .github/workflows/     # CI/CD pipelines (GitHub Actions)
├── .githooks/             # commit-msg hook (conventional-commit enforcement)
├── .cmake/                # CMake helper modules (config, eigen, testing, mpi, ...)
├── apps/                  # Main application executable (PQ.cpp)
├── benchmarks/            # Google Benchmark suite + benchmarks/perf (CI perf gate)
├── config/                # licenseHeader.txt (mandatory GPL header template)
├── docs/                  # Sphinx (docs/sphinx) + Doxygen (docs/doxygen) documentation
├── examples/              # Example simulation input files
├── external/              # Git submodules (googletest, mstd); Eigen via FetchContent
├── include/               # Header files (.hpp), mirrors src/ module layout
├── integration_tests/     # pytest-based end-to-end tests
├── scripts/               # Build/Singularity/conda scripts, changelog tooling + its tests
├── src/                   # Implementation files (.cpp), one CMakeLists.txt per module
├── tests/                 # GoogleTest unit tests, mirrors src/ + include/ layout
├── CMakeLists.txt         # Root build configuration
├── CHANGELOG.md           # User-facing release notes
├── DEV-CHANGELOG.md       # Developer/technical record (generated from commits at release)
└── cliff.toml             # git-cliff config used to render DEV-CHANGELOG.md
```

---

## Build System

**Requirements:**
- CMake >= 3.20
- GCC >= 13.0 (C++23), other compilers are untested/unsupported
- Eigen (fetched automatically via CMake `FetchContent`)

**Build commands:**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j<#procs>
# binary is produced under build/apps
```

**Key CMake options** (`.cmake/config.cmake`):

| Option                        | Default | Description                                    |
| ----------------------------- | ------- | ----------------------------------------------- |
| `BUILD_SHARED_LIBS`           | `ON`    | Build using shared libraries                    |
| `BUILD_WITH_TESTS`            | `ON`    | Build GoogleTest unit tests                     |
| `BUILD_WITH_MPI`              | `OFF`   | Build with MPI (needed for Ring Polymer MD)     |
| `BUILD_WITH_PYTHON_BINDINGS`  | `OFF`   | Build Python bindings                           |
| `BUILD_WITH_IWYU`             | `OFF`   | Build with include-what-you-use                 |
| `BUILD_WITH_GCOVR`            | `OFF`   | Build with gcovr coverage                       |
| `BUILD_WITH_DOCS`             | `ON`    | Build documentation                             |
| `BUILD_WITH_BENCHMARKING`     | `OFF`   | Build the Google Benchmark suite                |
| `BUILD_WITH_ASE`              | `ON`    | Build with ASE (QM runner integration)          |
| `BUILD_WITH_SINGULARITY`      | `OFF`   | Build with Singularity                          |
| `BUILD_WITH_LTO`              | `OFF`   | Release build with link-time optimization       |
| `BUILD_WITH_NATIVE`           | `ON`    | `-march=native` for Release (turn OFF for CI/portable builds) |
| `BUILD_WITH_PERF_BENCH`       | `OFF`   | Build the fixed-work perf-regression benchmark (`benchmarks/perf`), used by CI's callgrind gate |

If `ccache` and/or `mold`/`ld.lld` are available on `PATH`, CMake wires them in automatically to speed up rebuilds/linking.

---

## Source Code Architecture

Both `src/` and `include/` share the same module layout (each module has its own `CMakeLists.txt`):

**Simulation Engine:** `engine/` (orchestration), `integrator/`, `thermostat/`, `manostat/`

**Physical Models:** `forceField/` (MM bonded), `intraNonBonded/` (MM intramolecular non-bonded), `potential/` (MM intermolecular non-bonded), `QM/` (interface to QM runner programs)

**System Setup:** `input/`, `setup/`, `simulationBox/` (cell/molecule/atom handling), `connectivity/` (topology/bonding)

**Data Management:** `box/` (geometry, PBC), `physicalData/` (constants, unit conversions), `output/`

**Computational Infrastructure:** `linearAlgebra/`, `utilities/`, `kernels/`, `timings/`

**Advanced Features:** `constraints/` (SHAKE/RATTLE/M-SHAKE), `maxwellBoltzmann/`, `resetKinetics/`, `virial/`, `mpi/`, `python/`, `opt/`

**Design Patterns:** `concepts/` (C++23 concepts for template constraints), `exceptions/`, `settings/`, `config/`

---

## Naming Conventions

| Element         | Convention                        | Example                          |
| --------------- | ---------------------------------- | --------------------------------- |
| Header guards   | `_NAME_HPP_`                       | `_SIMULATION_BOX_HPP_`           |
| Namespaces      | lowercase, matches directory name  | `simulationBox`, `constraints`   |
| Doc comments    | Doxygen (`/** @class ... */`)      | see any file under `include/`    |

---

## Code Formatting

Formatting is enforced by **clang-format 20.1.3** (Google style base, customized in `.clang-format`):

- Indent: **4 spaces**, column limit **80**
- Brace style: **Allman**
- `BinPackArguments`/`BinPackParameters`: false
- Line endings: LF

CI (`clang_format.yml`) only checks **lines changed in the PR** via `git-clang-format` against the PR base, for `apps`, `benchmarks`, `include`, `src`, `tests` (excludes `external/`). Run locally before committing:

```bash
clang-format -i <file>
```

---

## License Header (Mandatory)

Every `.cpp`, `.hpp`, `.c`, `.h` file under `src/`, `tests/`, `include/`, `apps/`, `benchmarks/` must start with the **exact** GPL header from `config/licenseHeader.txt`. CI (`license_check.yml`) does a byte-for-byte comparison and fails on any mismatch. To add/fix headers:

```bash
bash scripts/addLicense.sh
```

(Not applicable to Markdown/docs files such as this one.)

---

## Testing

**Unit tests** use GoogleTest (submodule at `external/googletest`), built when `BUILD_WITH_TESTS=ON` (default).

```
tests/
├── src/        # mirrors src/ module layout
├── include/    # mirrors include/ module layout (test macros etc.)
└── data/       # test fixture files (input/topology/parameter readers, etc.)
```

Run with `make test` or `ctest --output-on-failure` from the build directory. When adding features, add corresponding tests mirroring the source module structure.

**Integration tests** are `pytest`-based in `integration_tests/` (see `pytest.ini`). They require `pytest`, `pytest-cov`, `pqanalysis`, `ase`, `pyscf`, plus `dftbplus` and `xtb` (installed via conda in CI). Run with `pytest integration_tests`.

**Performance regression gate** (`perf.yml`): for PRs touching `src/`, `include/`, `apps/`, `benchmarks/`, or build config, CI builds `benchmarks/perf` (fixed-work benchmarks, `BUILD_WITH_PERF_BENCH=ON`) for both the PR and its base branch, runs them under `callgrind`, and fails if instruction counts regress beyond threshold (`scripts/perf_gate.sh`). Results are posted as a PR comment.

---

## CI/CD Pipelines

All pipelines are in `.github/workflows/`:

| Workflow                           | Trigger                     | Purpose                                              |
| ----------------------------------- | ---------------------------- | ----------------------------------------------------- |
| `ci_build.yml`                      | PRs (any branch), push to `dev`/`main` | Linux (x86_64 + arm) build + `ctest` + coverage + integration tests + benchmark smoke test + static-LTO build + MPI build |
| `clang_format.yml`                   | PRs to `main`/`dev`          | `git-clang-format` on changed C/C++ lines             |
| `license_check.yml`                 | PRs (any branch), push to `main`/`master` | Exact GPL header check on all C/C++ files       |
| `perf.yml`                           | PRs to `main`/`dev`          | Callgrind instruction-count regression gate           |
| `check-pr-for-release-version.yml`  | PRs to `main`                | Requires `Release-x.y.z` in title/body; validates changelog tooling and curated user release notes |
| `create-tag.yml`                    | Manual                       | Create a release tag                                  |
| `daily_ci.yml`                      | Schedule                     | Daily build/test                                       |
| `jekyll-gh-pages.yml`               | Push                         | Publish documentation site                             |
| `pr_request.yml`                    | PRs to `main`/`dev`          | Fails if the target branch has moved since the PR was opened (must be rebased/updated) |

---

## Changelogs

- **`CHANGELOG.md`** — user-facing, curated by hand, one bullet per notable change.
- **`DEV-CHANGELOG.md`** — developer/technical record, generated from conventional commits via `git-cliff` (`cliff.toml`) at release time.

**Regular feature/bugfix PRs (targeting `dev`) do not edit either changelog file.** Changelog curation happens only in the release PR from `dev` to `main`, validated by `scripts/update_changelog.py --check` (see `check-pr-for-release-version.yml`). Do not add changelog entries as part of routine PRs.

---

## External Dependencies (Submodules)

Located in `external/` (`.gitmodules`):

- `googletest` — unit testing framework
- `mstd` — internal C++ standard library extensions

Eigen is fetched automatically via CMake `FetchContent` (not a submodule). Clone with `--recurse-submodules` or run `git submodule update --init --recursive` after cloning; the CMake `testing` module does this automatically if needed.

---

## Documentation

- **Sphinx** (`docs/sphinx/`, `.rst` sources) — user/developer guide, built with `make html` inside `docs/sphinx/`.
- **Doxygen** (`docs/doxygen/`) — API reference generated from source comments; build with `cmake .. -DBUILD_WITH_DOC=ON && make docs`.

---

## Development Environment

Singularity/conda definition files and helper scripts live in `scripts/`: `PQ.def`, `PQ_conda.def`, `PQ_openmpi.def`, `conda_build.sh`.

---

## Contribution Checklist

Before submitting any change:

- [ ] Branch created from `dev` (not `main`), named `feature/...` or `bugfix/...`
- [ ] PR targets `dev` (not `main`), unless this is an explicit release PR
- [ ] `.githooks/commit-msg` installed and all commit subjects use an approved prefix
- [ ] Code formatted with `clang-format` (Google style, 4-space indent, 80 cols, Allman braces)
- [ ] Every new/modified `.cpp`/`.hpp`/`.c`/`.h` file has the exact license header (`scripts/addLicense.sh`)
- [ ] Header guards follow `_NAME_HPP_`; namespaces match directory names
- [ ] Tests added/updated in `tests/` mirroring the changed `src/`/`include/` module
- [ ] No changes to `CHANGELOG.md` or `DEV-CHANGELOG.md` in routine PRs
- [ ] Git submodules (`external/googletest`, `external/mstd`) not accidentally modified
- [ ] No `-march=native`/native-only assumptions introduced into CI-relevant build paths (respect `BUILD_WITH_NATIVE`)
