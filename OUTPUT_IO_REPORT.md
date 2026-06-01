# PQ output system — performance investigation

For @97gamjak. All numbers from a Linux x86_64 box, GCC-13, Release build
(`-O3`, `-DBUILD_WITH_NATIVE=Off`), `examples/h2o_mm` workload (6000 atoms,
GUFF, cell-list, 9 Å cutoff). Three identical runs averaged per row.
Total times include compute; the meaningful number is the heavy-vs-none
delta, which is the output system's cost.

## TL;DR

| variant | heavy I/O (output every step) | no traj output | Δ (I/O cost) |
|---|---|---|---|
| **baseline** (current `dev`) | 98.75 s | 85.11 s | **13.64 s** |
| A1: drop per-write `std::flush` | 98.46 s | 85.16 s | 13.30 s (noise) |
| A2: 256 KB `ofstream` buffer | 98.59 s | 85.02 s | 13.57 s (noise) |
| **A3: one `std::format` per atom line** | 97.10 s | 85.25 s | **11.85 s** |

Workload is `nstep=500`, `output_freq=1` (worst-case I/O) vs
`output_freq=99999` (no trajectory output). Each output cycle writes
~12 separate ASCII files.

Only **A3** moves the needle — saves 1.65 s (~13 % of the I/O cost), one
line per writer, ASCII output unchanged. **A1 and A2 are no-ops** —
explained below. The other ~12 s of I/O cost is the `std::format` /
`ostringstream` / `vector<string>` machinery itself, not kernel I/O.

## What the experiments showed

### A1 — drop per-write `std::flush` — no measurable effect

The hot-path writers (`TrajectoryOutput::writeXyz/Velocities/Forces/Charges`,
`EnergyOutput`, `MomentumOutput`, `VirialOutput`, `StressOutput`,
`BoxOutput`) each end with `_fp << std::flush;`. Hypothesis: each flush
forces an extra `write(2)`. Removed all of them, measured: no change.

Reason: every writer builds the whole file in an `std::ostringstream`
first, then does `_fp << buffer.str()` — that already lands as one big
chunk in the `ofstream` userspace buffer, and the final flush at file
close commits it. The `std::flush` in between is essentially redundant
on Linux/ext4/SSD where the kernel handles writeback async anyway.

Keep the explicit `std::flush` in `LogOutput` (crash-recovery wants
progress on disk).

### A2 — 256 KB `ofstream` buffer — no measurable effect

Bumped the libstdc++ default ~4 KB output buffer to 256 KB via
`pubsetbuf` before `open()`. Hypothesis: fewer `write(2)` syscalls per
output cycle (~100 down to ~2 per file). Measured: no change.

Reason: the `ostringstream` buffer already holds the full file content
*in user space* before it touches the `ofstream`. The `ofstream` buffer
is effectively a small staging area between two userspace buffers, so
sizing it doesn't change the syscall count meaningfully — the whole
`buffer.str()` ends up going through `xsputn` in one or two passes
regardless. Don't bother.

### A3 — one `std::format` per atom line — saves ~1.7 s

Today each atom line in the trajectory/velocity/force writers does:

```cpp
buffer << std::format("{:<5}\t",   atom->getName());
buffer << std::format("{:15.8f}\t", pos[0]);
buffer << std::format("{:15.8f}\t", pos[1]);
buffer << std::format("{:15.8f}\n", pos[2]);
```

Collapsed to one call:

```cpp
buffer << std::format(
    "{:<5}\t{:15.8f}\t{:15.8f}\t{:15.8f}\n",
    atom->getName(), pos[0], pos[1], pos[2]);
```

Saves the `std::vformat` / parser / sink-iterator setup three times per
line. **ASCII output is bit-identical**. Worth shipping as a small PR.

### perf-record on the baseline — where the residual cost lives

`perf record -F 200 --call-graph=dwarf` on one heavy-output run, top
self-time entries (CPU-time, not wall):

| symbol | self % |
|---|---|
| `Potential::calculateSingleInteraction` | 30.5 % |
| `OrthorhombicBox::calcShiftVector` | 17.7 % |
| `Molecule::getAtomPosition` | 9.2 % |
| `PotentialCellList::calculateForces` | 9.0 % |
| `Atom::getPosition` | 8.3 % |
| `std::vformat` | 2.0 % |
| `__formatter_fp<char>::parse` | 0.9 % |
| `__formatter_fp<char>::format<double>` | 0.7 % |
| `__write_padded` | 0.7 % |
| `malloc` | 0.6 % |
| `__ostream_insert` | 0.5 % |
| `TrajectoryOutput::writeVelocities` | 0.04 % |
| `TrajectoryOutput::writeForces` | 0.03 % |
| `TrajectoryOutput::writeXyz` | 0.03 % |
| `TrajectoryOutput::writeCharges` | 0.02 % |

Two important things:

1. **75 % of CPU time is force computation**, not output. The output
   system is at most ~5–7 % even with the worst-case `output_freq=1`.
   For typical `output_freq=10` it drops to ~0.7 %, and for
   `output_freq=100` to ~0.07 %. So **for normal production runs the
   current code is fine**.
2. **The output cost that exists is concentrated in `std::format`
   machinery and dynamic-allocation churn**, not in the kernel I/O path.
   That's why A1 and A2 don't help.

## Where the bigger wins are (cost ↔ payoff)

### Drop `std::ostringstream`, write directly to `_fp` (Medium effort, ~3–5 s)

The `ostringstream → buffer.str() → _fp` round-trip allocates a fresh
`std::string` per write and copies the whole file body twice (once into
the ostringstream's internal buffer, once via `_fp << buffer.str()`).
The cleaner shape — and what most MD codes do — is to format directly
into the `ofstream`:

```cpp
std::format_to(std::ostreambuf_iterator<char>(_fp),
               "{:<5}\t{:15.8f}\t{:15.8f}\t{:15.8f}\n",
               atom->getName(), pos[0], pos[1], pos[2]);
```

Eliminates one allocation + one copy per output cycle and per writer.
Bit-identical ASCII output, no API change for downstream tools. Worth
trying as a follow-up to A3.

### Replace `std::format` with `std::to_chars` / `snprintf` (Medium effort, ~3–4 s)

`std::vformat` is feature-rich (and slow). For our format strings —
fixed-width float, fixed-width name — `std::to_chars(buf, end, x,
std::chars_format::fixed, 8)` is ~3–5× faster per call and zero
allocations. Code becomes uglier; ASCII output stays bit-identical (or
near; `to_chars` rounding matches IEEE round-to-nearest).

### Combine streams into one file per output cycle (Medium effort, ~1–2 s + smaller disk footprint)

Today an output cycle opens / writes / flushes **12** separate files
(`.xyz`, `.vel`, `.frc`, `.chrg`, `.rst`, `.vir`, `.str`, `.box`, `.en`,
`.ien`, `.info`, `.mom`). One combined `.traj` file with a small frame
header per stream removes 11 file handles' worth of syscalls and disk
seeks, and gives the kernel a single contiguous writeback target.

Downside: **PQAnalysis** would need a new reader. The current
`TrajectoryReader` only knows one-stream-per-file. The combined-file
reader is structurally simple but is new code.

### Binary trajectory format — the big lever for the read side (Medium effort, big payoff on Python)

The user-facing pain ("Python analysis takes too long") is mostly the
*read* side. PQAnalysis is pure ASCII (no `h5py` / `netCDF4` /
`MDAnalysis` / `np.fromfile` / `struct` anywhere — confirmed by
greping the package). The reader is two-pass (counts frames once, then
parses) and uses `sscanf`-via-Cython per atom line. ASCII float
parsing is the fundamental cost — typically **10–30× slower than
parsing the same numbers from binary** for trajectory-sized data.

Three viable backends, ordered by integration cost:

1. **Raw binary append-only stream**: per frame write a small header
   (`nAtoms`, time, box) and a `double[nAtoms][3]` blob. Smallest code,
   smallest dependency. Custom format → PQAnalysis needs a new reader,
   but it's ~30 lines (open, mmap, reshape). Roughly 5× smaller files
   than the current ASCII (`{:15.8f}` is 15 bytes vs 8 bytes binary),
   and the reader can use `np.fromfile` directly.
2. **HDF5 single-file with all streams**: positions/velocities/forces/
   charges/energies/box live as separate datasets in one `.h5` file.
   Adds `libhdf5` as a build dep but it's standard everywhere. Read
   side: `h5py.File(...).datasets`. Cleanest API; closest match to what
   AMBER and recent GROMACS analysis ecosystems expect.
3. **NetCDF (AMBER convention) or DCD / XTC**: maximum interoperability
   with existing MD analysis stacks (MDAnalysis, VMD, mdtraj read all
   three). More code on the writer side because these formats have
   strict layout rules; biggest payoff if you want to ditch PQAnalysis
   altogether for trajectory analysis.

Whichever path: **keep the ASCII writers as opt-in for debugging /
short reference runs**. The binary format becomes the production
default; ASCII stays available behind a `output_format = ascii;` knob.

### Async writer thread (High effort, eliminates I/O from the critical path)

The `ostringstream` build per writer is pure compute on the main thread
today. Hand the populated buffer to a worker thread that does the
actual `write()`. Compute thread blocks only when the queue is full.
With the queue sized for ~2 output cycles, the main loop runs at the
"no-output" speed (here ~85 s instead of ~99 s); the writer thread is
free to lag behind.

Caveat: requires careful lifecycle — must drain on shutdown / restart
file write, and the restart file should still be synchronous so
crash-recovery sees consistent state.

## Recommendation

Three PRs, in order:

1. **A3 (now)** — collapse the per-coordinate `std::format` calls into
   one call per atom line in `TrajectoryOutput`. One-file diff, ASCII
   bit-identical, ~13 % of the output cost. Small, no design call.
2. **Direct `format_to` + `to_chars`** — drop the `ostringstream`
   intermediate and replace `std::format` with `std::to_chars` in the
   trajectory writers. ~3–7 s savings combined, still ASCII
   bit-identical (or within last-bit rounding), still no downstream
   change. Medium PR.
3. **Binary trajectory option** — add an opt-in binary trajectory
   format (raw stream first; HDF5 if you're OK with the dep). Biggest
   payoff is on the PQAnalysis side, not PQ itself. Needs a coordinated
   PQAnalysis PR but the reader change is small (~30 lines for the raw
   format). I'd skip async-writer until after this — async on top of
   ASCII makes the design harder than async on top of binary.

The "combine all streams into one file" idea is good in the abstract
but is best done **together** with the binary format (HDF5 naturally
combines streams into one file). Doing it on top of ASCII just shifts
the parsing cost to a new file layout.

## Appendix: workload, environment, scripts

- Workload: `examples/h2o_mm`, edited to `nstep=500`, `output_freq=1`
  (heavy) or `output_freq=99999` (no trajectory output). 6000 atoms.
- Build: GCC-13, `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_NATIVE=Off
  -DBUILD_WITH_TESTS=Off -DBUILD_WITH_DOCS=Off`. No LTO.
- Host: Linux x86_64, 6 cores, SSD-backed `ext4`. Single PQ binary, no
  external load during measurements.
- Bench harness: `/tmp/io_bench/run_bench.sh` (in /tmp on the test
  host); patches: `patch_A1_no_flush.sh`, `patch_A2_bigger_buffer.sh`,
  `patch_A3_combined_format.sh`. perf profile: `profile_io.sh`. All
  reproducible.
- Numbers are wall-clock `/usr/bin/time -f %e`. Three runs averaged per
  row; the spread between runs is ~0.5–1 s, so anything below ~1 s
  delta is noise. The 1.65 s A3 win clears that threshold.
