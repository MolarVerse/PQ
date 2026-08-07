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
    static constexpr auto* PQ_FILE = "pq.ref";

    // Time Integrators
    static constexpr auto* VELOCITY_VERLET_FILE = "velocity_verlet.ref";

    // Thermostats and Manostats
    static constexpr auto* BERENDSEN_FILE            = "berendsen.ref";
    static constexpr auto* VELOCITY_RESCALING_FILE   = "velocity_rescaling.ref";
    static constexpr auto* NOSE_HOOVER_CHAIN_FILE    = "nose_hoover_chain.ref";
    static constexpr auto* LANGEVIN_FILE             = "langevin.ref";
    static constexpr auto* STOCHASTIC_RESCALING_FILE = "stochastic_rescaling.ref";

    // QM Programs
    static constexpr auto* DFTBPLUS_FILE  = "dftbplus.ref";
    static constexpr auto* GFN1_FILE      = "gfn1.ref";
    static constexpr auto* GFN2_FILE      = "gfn2.ref";
    static constexpr auto* IPEA1_FILE     = "ipea1.ref";
    static constexpr auto* PYSCF_FILE     = "pyscf.ref";
    static constexpr auto* TURBOMOLE_FILE = "turbomole.ref";
    static constexpr auto* MACEMP_FILE    = "macemp.ref";
    static constexpr auto* MACEOFF_FILE   = "maceoff.ref";
    static constexpr auto* FENNOL_FILE    = "fennol.ref";

    // Constraint Dynamics
    static constexpr auto* RATTLE_FILE = "rattle.ref";

    // clang-format on

}   // namespace references

#endif   // _REFERENCES_HPP_
