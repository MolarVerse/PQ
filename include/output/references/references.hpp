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
    static constexpr char _GFN1_FILE_[]      = "gfn1.ref";
    static constexpr char _GFN2_FILE_[]      = "gfn2.ref";
    static constexpr char _IPEA1_FILE_[]     = "ipea1.ref";
    static constexpr char _PYSCF_FILE_[]     = "pyscf.ref";
    static constexpr char _TURBOMOLE_FILE_[] = "turbomole.ref";
    static constexpr char _MACEMP_FILE_[]    = "macemp.ref";
    static constexpr char _MACEOFF_FILE_[]   = "maceoff.ref";

    // Constraint Dynamics
    static constexpr char _RATTLE_FILE_[] = "rattle.ref";

    // Water Models
    static constexpr char _SPC_FILE_[]       = "water_model_spc.ref";
    static constexpr char _SPC_E_FILE_[]     = "water_model_spce.ref";
    static constexpr char _SPC_FW_FILE_[]    = "water_model_spcfw.ref";
    static constexpr char _QSPC_FW_FILE_[]   = "water_model_qspcfw.ref";
    static constexpr char _SPC_DC_FILE_[]    = "water_model_spcdc.ref";
    static constexpr char _H2O_DC_FILE_[]    = "water_model_h2odc.ref";
    static constexpr char _TIP3P_FILE_[]     = "water_model_tip3p.ref";
    static constexpr char _OPC3_FILE_[]      = "water_model_opc3.ref";
    static constexpr char _SPC_MTR_FILE_[]   = "water_model_spcmtr.ref";
    static constexpr char _TIP3P_MTR_FILE_[] = "water_model_tip3pmtr.ref";

    // clang-format on

}   // namespace references

#endif   // _REFERENCES_HPP_