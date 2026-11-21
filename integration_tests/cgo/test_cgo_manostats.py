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

The manostat variants that the other cgo tests do not use (they use the
isotropic Berendsen manostat).

* Berendsen with `isotropy = anisotropic` and with the semi-isotropic `xy`:
  deterministic, compared with references (the box is part of every frame
  header, so it is compared too).
* Berendsen anisotropic with `fixed_axis = z`: the z axis must not move at all.
* Stochastic cell rescaling: it uses random numbers, so there is no reference.
  It runs together with a temperature ramp, because it reads the global target
  temperature that the ramp changes; the same seed must give the same run, and
  another seed another box.

The relaxation time of 0.1 ps (as in the cell list tests) makes the box move by
a few percent in the 10 steps, so a wrong coupling shows.
"""

import pathlib
import re

import pytest

from analysis import check_pq_output
from conftest import execute_pq

SOURCE = "run-01_NPT.in"
BERENDSEN = "manostat = berendsen; pressure = 1.013; p_relaxation = 1.0;"
THERMOSTAT = "thermostat = berendsen; temp = 998.15; t_relaxation = 0.1;"
OUTPUT_FILES = ("xyz", "vel", "force", "box")
INITIAL_BOX_LENGTH = 27.44205774


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
@pytest.mark.parametrize(
    "input_file, reference",
    [
        ("run-01_NPT_anisotropic.in", "ref_data_NPT_anisotropic"),
        ("run-01_NPT_xy.in", "ref_data_NPT_xy"),
    ],
    ids=["anisotropic", "semi_isotropic_xy"],
)
def test_cgo_berendsen_manostat_isotropy(
        test_with_data_dir, input_file, reference):
    _, stderr = execute_pq(input_file)

    assert stderr == b""

    _assert_finite()
    check_pq_output("cgo-mm-01", "cgo", reference)


def _assert_finite():
    for extension in OUTPUT_FILES + ("en",):
        text = pathlib.Path(f"cgo-mm-01.{extension}").read_text().lower()
        assert "nan" not in text, f"cgo-mm-01.{extension} contains NaN"


def _run(prefix, manostat, thermostat=THERMOSTAT, extra=""):
    """Run the NPT input with another manostat (and thermostat); returns the
    text of xyz, vel, force and box, and the energy columns that do not depend
    on the machine (all but the last two: momentum and loop time)."""
    text = pathlib.Path(SOURCE).read_text()
    text = re.sub(r"file_prefix\s*=\s*[^;]+", f"file_prefix = {prefix}", text)
    assert BERENDSEN in text and THERMOSTAT in text
    text = text.replace(BERENDSEN, f"{manostat} {extra}", 1)
    text = text.replace(THERMOSTAT, thermostat, 1)
    pathlib.Path(f"{prefix}.in").write_text(text)

    _, stderr = execute_pq(f"{prefix}.in")

    assert stderr == b""

    files = {
        extension: pathlib.Path(f"{prefix}.{extension}").read_text()
        for extension in OUTPUT_FILES
    }
    for extension, content in files.items():
        assert "nan" not in content.lower(), f"{prefix}.{extension} has NaN"
    energies = [
        line.split()[:-2]
        for line in pathlib.Path(f"{prefix}.en").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    return files, energies


def _box_lengths(box_text):
    """The three box lengths of every frame of a .box file."""
    return [
        [float(value) for value in line.split()[1:4]]
        for line in box_text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


ANISOTROPIC = (
    "manostat = berendsen; pressure = 1.013; p_relaxation = 0.1; "
    "isotropy = anisotropic;"
)


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
def test_cgo_anisotropic_manostat_keeps_a_fixed_axis(test_with_data_dir):
    fixed, _ = _run("fixed", ANISOTROPIC, extra="fixed_axis = z;")
    free, _ = _run("free", ANISOTROPIC)

    fixed_boxes = _box_lengths(fixed["box"])
    free_boxes = _box_lengths(free["box"])

    assert len(fixed_boxes) > 1
    for _, _, z_length in fixed_boxes:
        assert z_length == pytest.approx(INITIAL_BOX_LENGTH, abs=1e-8)
    assert any(abs(x_length - INITIAL_BOX_LENGTH) > 1e-3
               for x_length, _, _ in fixed_boxes), "x must move"
    assert any(abs(z_length - INITIAL_BOX_LENGTH) > 1e-3
               for _, _, z_length in free_boxes), "z moves if it is not fixed"
    assert fixed["box"] != free["box"]


STOCHASTIC = (
    "manostat = stochastic_rescaling; pressure = 1.013; p_relaxation = 0.1;"
)
RAMP = (
    "thermostat = berendsen; start_temp = 900.0; temp = 998.15; "
    "t_relaxation = 0.1; temp_ramp_steps = 8; temp_ramp_frequency = 2;"
)


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
def test_cgo_stochastic_manostat_follows_the_seed(test_with_data_dir):
    seeded = f"{STOCHASTIC} random_seed = "

    first = _run("first", seeded + "1;", RAMP)
    again = _run("again", seeded + "1;", RAMP)
    other = _run("other", seeded + "2;", RAMP)
    no_ramp = _run("noramp", seeded + "1;")

    assert first == again, "the same seed must give the same run"
    assert first[0]["box"] != other[0]["box"], "another seed, another box"
    assert first[0]["xyz"] != other[0]["xyz"]
    assert first[0]["box"] != no_ramp[0]["box"], "the ramp must act"
    assert len(_box_lengths(first[0]["box"])) > 1
    assert any(abs(length - INITIAL_BOX_LENGTH) > 1e-3
               for box in _box_lengths(first[0]["box"]) for length in box)


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
@pytest.mark.parametrize("cells", ["", "cell-number = 16;"],
                         ids=["default_cells", "16_cells"])
def test_cgo_anisotropic_manostat_with_the_cell_list(test_with_data_dir, cells):
    """The three box lengths differ and move differently: the cell list must
    follow, and give the brute-force result."""
    text = pathlib.Path("run-01_NPT_anisotropic.in").read_text()
    text = text.replace(
        "integrator = v-verlet;",
        f"integrator = v-verlet; cell-list = on; {cells}", 1)
    pathlib.Path("cell_list.in").write_text(text)

    _, stderr = execute_pq("cell_list.in")

    assert stderr == b""

    _assert_finite()
    check_pq_output("cgo-mm-01", "cgo", "ref_data_NPT_anisotropic")
