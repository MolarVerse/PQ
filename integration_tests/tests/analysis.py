import numpy as np

from PQAnalysis.io import read_trajectory, EnergyFileReader


def check_pq_output(base_name, folder_name):

    ref_path = "../tests/" + folder_name + "/ref_data/"

    traj = read_trajectory(base_name + ".xyz")
    ref_traj = read_trajectory(ref_path + base_name + ".xyz")

    assert traj == ref_traj

    traj = read_trajectory(base_name + ".vel", traj_format="vel")
    ref_traj = read_trajectory(
        ref_path + base_name + ".vel", traj_format="vel")

    assert traj == ref_traj

    traj = read_trajectory(base_name + ".force", traj_format="force")
    ref_traj = read_trajectory(
        ref_path + base_name + ".force", traj_format="force")

    assert traj == ref_traj

    traj = read_trajectory(base_name + ".chrg", traj_format="charge")
    ref_traj = read_trajectory(
        ref_path + base_name + ".chrg", traj_format="charge")

    assert traj == ref_traj

    reader = EnergyFileReader(base_name + ".en")
    ref_reader = EnergyFileReader(
        ref_path + base_name + ".en")

    energy = reader.read()
    ref_energy = ref_reader.read()

    assert np.allclose(energy.data[:-1], ref_energy.data[:-1])

    assert energy.info_given
    assert energy.info == ref_energy.info
