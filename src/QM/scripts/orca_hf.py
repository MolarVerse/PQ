#!/usr/bin/env python3
"""PQ ORCA HF runner: pure QM or polarized QM/MM.

Arguments: charge reset_orbitals use_pointcharges template pointcharge_file.
Keep orca_scf.py and orca_common.py beside this script when copying it.
"""

import sys

sys.dont_write_bytecode = True

from orca_scf import cli

if __name__ == "__main__":
    cli("HF")
