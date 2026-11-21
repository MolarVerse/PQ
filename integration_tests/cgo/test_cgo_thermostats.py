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

The thermostats that the other cgo tests do not use (they use Berendsen).

* Nose-Hoover chain: deterministic, so it is compared with a reference. It runs
  with a temperature ramp, because the chain keeps its own state (positions
  and velocities of the thermostat variables) that a ramp moves.
* Velocity rescaling (Bussi) and Langevin: stochastic. A reference would depend
  on the random number generator of the standard library, so the tests check
  what must hold on any toolchain: the same seed gives the same run bit for
  bit, a different seed gives another run, and the thermostat changes the run
  compared with plain NVE.
"""

import pathlib
import re

import pytest

from analysis import check_pq_output
from conftest import execute_pq

SOURCE = "run-01.in"
OUTPUT_FILES = ("xyz", "vel", "force")

STOCHASTIC_THERMOSTATS = {
    "velocity_rescaling": (
        "thermostat = velocity_rescaling; temp = 500.0; t_relaxation = 0.05;"
    ),
    "langevin": "thermostat = langevin; temp = 500.0; friction = 5.0;",
}


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
def test_cgo_nose_hoover_ramp(test_with_data_dir):
    _, stderr = execute_pq("run-01_nose_hoover.in")

    assert stderr == b""

    # a chain that is driven too hard blows up to NaN; never accept that
    for extension in OUTPUT_FILES + ("en",):
        text = pathlib.Path(f"cgo-mm-01.{extension}").read_text().lower()
        assert "nan" not in text, f"cgo-mm-01.{extension} contains NaN"

    check_pq_output("cgo-mm-01", "cgo", "ref_data_nose_hoover")


def _run(prefix, thermostat, seed):
    """Run run-01.in with the thermostat keywords and the seed; return the
    trajectory text (xyz, vel, force) and the energy columns that do not
    depend on the machine (all but the last two, which are the momentum and
    the loop time)."""
    text = pathlib.Path(SOURCE).read_text()
    text = re.sub(r"file_prefix\s*=\s*[^;]+", f"file_prefix = {prefix}", text)
    text = text.replace(
        "integrator = v-verlet;",
        f"integrator = v-verlet; {thermostat} random_seed = {seed};",
        1)
    pathlib.Path(f"{prefix}.in").write_text(text)

    _, stderr = execute_pq(f"{prefix}.in")

    assert stderr == b""

    files = {
        extension: pathlib.Path(f"{prefix}.{extension}").read_text()
        for extension in OUTPUT_FILES
    }
    energies = [
        line.split()[:-2]
        for line in pathlib.Path(f"{prefix}.en").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    return files, energies


@pytest.mark.parametrize("example_dir", ["cgo/"], indirect=False)
@pytest.mark.parametrize("thermostat", sorted(STOCHASTIC_THERMOSTATS))
def test_cgo_stochastic_thermostat_follows_the_seed(
        test_with_data_dir, thermostat):
    keywords = STOCHASTIC_THERMOSTATS[thermostat]

    first = _run("first", keywords, seed=1)
    again = _run("again", keywords, seed=1)
    other = _run("other", keywords, seed=2)
    nve = _run("nve", "", seed=1)

    assert first == again, "the same seed must give the same run"
    assert first[0] != other[0], "another seed must give another run"
    assert first[1] != other[1], "another seed must give other energies"
    assert first[0] != nve[0], "the thermostat must change the run"
    assert first[1] != nve[1], "the thermostat must change the energies"
