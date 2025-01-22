/*****************************************************************************
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
******************************************************************************/

#ifndef _REFERENCES_HPP_

#define _REFERENCES_HPP_

namespace references
{
    // clang-format off
    // PQ Software
    static constexpr char _PQ_FILE_[] = "pq.ref";

    // Time Integrators
    static constexpr char _VELOCITY_VERLET_FILE_[] = "velocity_verlet.ref";

    // Thermostats and Manostats
    static constexpr char _BERENDSEN_FILE_[]            = "berendsen.ref";
    static constexpr char _VELOCITY_RESCALING_FILE_[]   = "velocity_rescaling.ref";
    static constexpr char _NOSE_HOOVER_CHAIN_FILE_[]    = "nose_hoover_chain.ref";
    static constexpr char _LANGEVIN_FILE_[]             = "langevin.ref";
    static constexpr char _STOCHASTIC_RESCALING_FILE_[] = "stochastic_rescaling.ref";

    // QM Programs
    static constexpr char _DFTBPLUS_FILE_[]  = "dftbplus.ref";
    static constexpr char _PYSCF_FILE_[]     = "pyscf.ref";
    static constexpr char _TURBOMOLE_FILE_[] = "turbomole.ref";
    static constexpr char _MACEMP_FILE_[]    = "macemp.ref";
    static constexpr char _MACEOFF_FILE_[]   = "maceoff.ref";
    static constexpr char _FAIRCHEM_FILE[]   = "fairchem.ref";

    // Constraint Dynamics
    static constexpr char _RATTLE_FILE_[] = "rattle.ref";

    // clang-format on

}   // namespace references

#endif   // _REFERENCES_HPP_