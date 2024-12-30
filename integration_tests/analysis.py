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

import numpy as np

from PQAnalysis.io import read_trajectory, EnergyFileReader


def check_pq_output(base_name, folder_name, ref_folder="ref_data"):

    ref_path = "../" + folder_name + "/" + ref_folder + "/"

    traj = read_trajectory(base_name + ".xyz")
    ref_traj = read_trajectory(ref_path + base_name + ".xyz")

    assert traj.isclose(ref_traj)

    traj = read_trajectory(base_name + ".vel", traj_format="vel")
    ref_traj = read_trajectory(
        ref_path + base_name + ".vel", traj_format="vel")

    assert traj.isclose(ref_traj)

    traj = read_trajectory(base_name + ".force", traj_format="force")
    ref_traj = read_trajectory(
        ref_path + base_name + ".force", traj_format="force")

    assert traj.isclose(ref_traj)

    traj = read_trajectory(base_name + ".chrg", traj_format="charge")
    ref_traj = read_trajectory(
        ref_path + base_name + ".chrg", traj_format="charge")

    assert traj.isclose(ref_traj)

    reader = EnergyFileReader(base_name + ".en")
    ref_reader = EnergyFileReader(
        ref_path + base_name + ".en")

    energy = reader.read()
    ref_energy = ref_reader.read()

    assert np.allclose(energy.data[:-2], ref_energy.data[:-2], atol=1e-6)

    assert energy.info_given
    assert energy.info == ref_energy.info
