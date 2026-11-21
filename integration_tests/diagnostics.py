"""
*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
*****************************************************************************

Messages for failing integration tests: what went wrong, not just that
something did. Pure Python (no pqanalysis, no numpy), so it is unit tested in
scripts/tests/test_integration_diagnostics.py.
"""

COLUMNS = ("x", "y", "z")
MAX_REPORTED_HEADERS = 3


def format_process_failure(command, returncode, stdout, stderr,
                           stdout_lines=40, stderr_lines=60):
    """The message for a PQ run that exited with a non-zero status.

    The end of stdout is where PQ prints its error; stderr is shown whole
    (up to stderr_lines).
    """
    def tail(data, count):
        lines = [line for line in data.decode(errors="replace").splitlines()
                 if line.strip()]
        shown = lines[-count:]
        if len(lines) > count:
            shown.insert(0, f"... ({len(lines) - count} earlier lines left out)")
        return "\n".join(shown) if shown else "(empty)"

    return (
        f"PQ failed with exit status {returncode}: {' '.join(command)}\n"
        f"--- end of stdout ---\n{tail(stdout, stdout_lines)}\n"
        f"--- stderr ---\n{tail(stderr, stderr_lines)}"
    )


def read_frames(path):
    """The frames of a PQ trajectory file.

    Returns a list of (header tokens, atoms), atoms being (name, values) with
    the values as floats. Works for .xyz, .vel, .force and .chrg files.
    """
    with open(path, encoding="utf-8") as stream:
        lines = stream.read().split("\n")

    frames = []
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        header = lines[index].split()
        count = int(header[0])
        atoms = []

        for line in lines[index + 2:index + 2 + count]:
            tokens = line.split()
            atoms.append((tokens[0], [float(token) for token in tokens[1:]]))

        frames.append((header, atoms))
        index += 2 + count

    return frames


def _excess(value, reference, rtol, atol):
    """|value - reference| in units of the allowed deviation (> 1 fails).

    A NaN on either side is never close (as in isclose), so it counts as
    infinitely far."""
    if value != value or reference != reference:
        return float("inf")

    allowed = atol + rtol * abs(reference)
    deviation = abs(value - reference)

    if allowed == 0.0:
        return 0.0 if deviation == 0.0 else float("inf")

    return deviation / allowed


def describe_trajectory_difference(path, ref_path, rtol, atol=0.0):
    """What differs between a trajectory file and its reference.

    Compares like isclose: |value - reference| <= atol + rtol * |reference|.
    The text says which file, and where the worst value is (frame, atom, column,
    both values), how many values are out of tolerance and which headers (box)
    differ.
    """
    frames = read_frames(path)
    ref_frames = read_frames(ref_path)
    lines = [f"{path} differs from the reference {ref_path} "
             f"(rtol={rtol:g}, atol={atol:g})"]

    if len(frames) != len(ref_frames):
        lines.append(f"  number of frames: {len(frames)}, reference has "
                     f"{len(ref_frames)}")
        return "\n".join(lines)

    outside, total = 0, 0
    worst = None
    header_differences = []

    for frame_index, ((header, atoms), (ref_header, ref_atoms)) in enumerate(
            zip(frames, ref_frames)):
        if header[0] != ref_header[0] or len(atoms) != len(ref_atoms):
            lines.append(f"  frame {frame_index}: {len(atoms)} atoms, "
                         f"reference has {len(ref_atoms)}")
            return "\n".join(lines)

        if header != ref_header:
            header_differences.append((frame_index, header, ref_header))

        for atom_index, ((name, values), (ref_name, ref_values)) in enumerate(
                zip(atoms, ref_atoms)):
            if name != ref_name or len(values) != len(ref_values):
                lines.append(f"  frame {frame_index}, atom {atom_index}: "
                             f"'{name}' with {len(values)} values, reference "
                             f"'{ref_name}' with {len(ref_values)}")
                return "\n".join(lines)

            for column, (value, ref_value) in enumerate(
                    zip(values, ref_values)):
                total += 1
                excess = _excess(value, ref_value, rtol, atol)
                if excess > 1.0:
                    outside += 1
                if worst is None or excess > worst[0]:
                    worst = (excess, frame_index, atom_index, name, column,
                             value, ref_value)

    lines.append(f"  {outside} of {total} values are out of tolerance")

    if worst is not None and worst[0] > 1.0:
        _, frame_index, atom_index, name, column, value, ref_value = worst
        label = COLUMNS[column] if len(frames[0][1][0][1]) == 3 else column
        difference = abs(value - ref_value)
        relative = f"{difference / abs(ref_value):.3g}" if ref_value else "n/a"
        lines.append(
            f"  worst: frame {frame_index}, atom {atom_index} ({name}), "
            f"column {label}: {value!r} vs reference {ref_value!r}, "
            f"difference {difference:.3g} (relative {relative})"
        )

    for frame_index, header, ref_header in header_differences[
            :MAX_REPORTED_HEADERS]:
        lines.append(f"  header of frame {frame_index}: "
                     f"'{' '.join(header)}' vs reference "
                     f"'{' '.join(ref_header)}'")
    if len(header_differences) > MAX_REPORTED_HEADERS:
        lines.append(f"  ... and {len(header_differences) - MAX_REPORTED_HEADERS}"
                     f" more headers differ")

    return "\n".join(lines)


def describe_table_difference(name, rows, ref_rows, rtol, atol):
    """What differs between two tables of numbers (the energy file).

    rows and ref_rows are lists of rows; the comparison is like numpy's
    allclose: |value - reference| <= atol + rtol * |reference|.
    """
    lines = [f"{name} differs from the reference (rtol={rtol:g}, atol={atol:g})"]

    if len(rows) != len(ref_rows):
        lines.append(f"  number of rows: {len(rows)}, reference has "
                     f"{len(ref_rows)}")
        return "\n".join(lines)

    outside, total = 0, 0
    worst = None

    for row_index, (row, ref_row) in enumerate(zip(rows, ref_rows)):
        if len(row) != len(ref_row):
            lines.append(f"  row {row_index}: {len(row)} columns, reference "
                         f"has {len(ref_row)}")
            return "\n".join(lines)

        for column, (value, ref_value) in enumerate(zip(row, ref_row)):
            total += 1
            excess = _excess(value, ref_value, rtol, atol)
            if excess > 1.0:
                outside += 1
            if worst is None or excess > worst[0]:
                worst = (excess, row_index, column, value, ref_value)

    lines.append(f"  {outside} of {total} values are out of tolerance")

    if worst is not None and worst[0] > 1.0:
        _, row_index, column, value, ref_value = worst
        lines.append(
            f"  worst: row {row_index}, column {column}: {value!r} vs "
            f"reference {ref_value!r}, difference "
            f"{abs(value - ref_value):.3g}"
        )

    return "\n".join(lines)
