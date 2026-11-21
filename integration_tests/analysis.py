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
"""

from pathlib import Path

import numpy as np

from PQAnalysis.io import read_trajectory, EnergyFileReader

from diagnostics import (describe_table_difference,
                         describe_trajectory_difference)

# Relative tolerance for comparing trajectories against the reference data.
# The output has 9 significant digits, so the pqanalysis default of 1e-9 fails
# on a one-digit difference in the last printed digit (floating-point roundoff
# from a different order of unit conversions).
TRAJ_RTOL = 1e-5

# Absolute tolerance for the forces of tests that use dftb+: its forces carry a
# noise of about 1e-5 kcal/mol/Angstrom (SCC convergence), which a purely
# relative tolerance turns into a failure for components close to zero. Other
# tests compare forces with the relative tolerance only.
DFTB_FORCE_ATOL = 1e-4


def _check_trajectory(base_name, ref_path, extension, rtol, atol=0.0,
                      traj_format=None):
    name = base_name + "." + extension
    ref_name = str(ref_path / name)
    options = {} if traj_format is None else {"traj_format": traj_format}

    traj = read_trajectory(name, **options)
    ref_traj = read_trajectory(ref_name, **options)

    if not traj.isclose(ref_traj, rtol=rtol, atol=atol):
        raise AssertionError(
            describe_trajectory_difference(name, ref_name, rtol, atol))


def check_pq_output(
    base_name,
    folder_name,
    ref_folder="ref_data",
    force_atol=0.0,
):

    ref_path = Path(ref_folder)

    _check_trajectory(base_name, ref_path, "xyz", TRAJ_RTOL)
    _check_trajectory(base_name, ref_path, "vel", TRAJ_RTOL,
                      traj_format="vel")
    _check_trajectory(base_name, ref_path, "force", TRAJ_RTOL, force_atol,
                      traj_format="force")
    _check_trajectory(base_name, ref_path, "chrg", TRAJ_RTOL,
                      traj_format="charge")

    reader = EnergyFileReader(base_name + ".en")
    ref_reader = EnergyFileReader(
        str(ref_path / (base_name + ".en")))

    energy = reader.read()
    ref_energy = ref_reader.read()

    if not np.allclose(energy.data[:-2], ref_energy.data[:-2]):
        raise AssertionError(describe_table_difference(
            base_name + ".en",
            np.asarray(energy.data[:-2]).T.tolist(),
            np.asarray(ref_energy.data[:-2]).T.tolist(),
            rtol=1e-5, atol=1e-8))

    assert energy.info_given, f"{base_name}.en has no info block"
    assert energy.info == ref_energy.info, (
        f"the column description of {base_name}.en differs from the "
        f"reference:\n  {energy.info}\n  reference: {ref_energy.info}")
