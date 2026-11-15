#!/usr/bin/env python3
"""PQ Turbomole driver, RI-CC2/RI-MP2 (dscf, ricc2); arguments: charge run_define use_pc template pc_file."""

import sys

sys.dont_write_bytecode = True

from turbomole_common import RICC2, main

if __name__ == "__main__":
    main(RICC2)
