#!/usr/bin/env python3
"""PQ Turbomole driver, RI-DFT (ridft, rdgrad); arguments: charge run_define use_pc template pc_file."""

import sys

sys.dont_write_bytecode = True

from turbomole_common import RIDFT, main

if __name__ == "__main__":
    main(RIDFT)
