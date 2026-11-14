#!/usr/bin/env python3
"""PQ Turbomole driver, HF/DFT (dscf, grad); arguments: charge run_define use_pc template pc_file."""

import sys

sys.dont_write_bytecode = True

from turbomole_common import HF_DFT, main

if __name__ == "__main__":
    main(HF_DFT)
