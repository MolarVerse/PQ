from pathlib import Path


number_of_atoms = int(Path("coords.xyz").read_text().splitlines()[0])

Path("qm_forces").write_text(
    "0.0\n" + "0.0 0.0 0.0\n" * number_of_atoms
)
Path("qm_charges").write_text(
    "".join(f"{index} 0.0\n" for index in range(1, number_of_atoms + 1))
)
