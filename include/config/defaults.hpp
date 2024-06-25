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

#ifndef _DEFAULTS_HPP_

#define _DEFAULTS_HPP_

#include <string>

/**
 * @brief namespace containing all default values
 *
 */
namespace defaults
{
    // clang-format off
    static constexpr char   _MOLDESCRIPTOR_FILE_DEFAULT_[] = "moldescriptor.dat";
    static constexpr char   _GUFF_FILE_DEFAULT_[]      = "guff.dat";
    static constexpr size_t _NUMBER_OF_GUFF_ENTRIES_       = 28;

    static constexpr char _RESTART_FILE_DEFAULT_[]  = "default.rst";
    static constexpr char _ENERGY_FILE_DEFAULT_[]   = "default.en";
    static constexpr char _INSTEN_FILE_DEFAULT_[]   = "default.instant_en";
    static constexpr char _MOMENTUM_FILE_DEFAULT_[] = "default.mom";
    static constexpr char _TRAJ_FILE_DEFAULT_[]     = "default.xyz";
    static constexpr char _VEL_FILE_DEFAULT_[]      = "default.vel";
    static constexpr char _FORCE_FILE_DEFAULT_[]    = "default.force";
    static constexpr char _CHARGE_FILE_DEFAULT_[]   = "default.charge";
    static constexpr char _LOG_FILE_DEFAULT_[]      = "default.out";
    static constexpr char _INFO_FILE_DEFAULT_[]     = "default.info";
    static constexpr char _VIRIAL_FILE_DEFAULT_[]   = "default.vir";
    static constexpr char _STRESS_FILE_DEFAULT_[]   = "default.stress";
    static constexpr char _BOX_FILE_DEFAULT_[]      = "default.box";
    static constexpr char _OPT_FILE_DEFAULT_[]      = "default.opt";
    static constexpr char _TIMINGS_FILE_DEFAULT_[]  = "default.timings";

    static constexpr char _RPMD_RST_FILE_DEFAULT_[]    = "default.rpmd.rst";
    static constexpr char _RPMD_TRAJ_FILE_DEFAULT_[]   = "default.rpmd.xyz";
    static constexpr char _RPMD_VEL_FILE_DEFAULT_[]    = "default.rpmd.vel";
    static constexpr char _RPMD_FORCE_FILE_DEFAULT_[]  = "default.rpmd.force";
    static constexpr char _RPMD_CHARGE_FILE_DEFAULT_[] = "default.rpmd.charge";
    static constexpr char _RPMD_ENERGY_FILE_DEFAULT_[] = "default.rpmd.en";

    static constexpr double _COULOMB_CUT_OFF_DEFAULT_           = 12.5;   // in Angstrom
    static constexpr double _SCALE_14_COULOMB_DEFAULT_          = 1.0;
    static constexpr double _SCALE_14_VAN_DER_WAALS_DEFAULT_    = 1.0;
    static constexpr double _WOLF_PARAM_DEFAULT_            = 0.25;     // TODO: add unit
    static constexpr char   _COULOMB_LR_TYPE_DEFAULT_[] = "none";   // default no coulomb long range correction
    static constexpr char   _NON_COULOMB_TYPE_DEFAULT_[]        = "guff";   // default is guff

    static constexpr bool   _CONSTRAINTS_ACTIVE_DEFAULT_ = false;
    static constexpr size_t _SHAKE_MAX_ITER_DEFAULT_     = 20;
    static constexpr size_t _RATTLE_MAX_ITER_DEFAULT_    = 20;
    static constexpr double _SHAKE_TOLERANCE_DEFAULT_    = 1e-8;
    static constexpr double _RATTLE_TOLERANCE_DEFAULT_   = 1e-8 * 1e12;

    static constexpr bool   _CELL_LIST_IS_ACTIVE_DEFAULT_ = false;   // default is brute force routine
    static constexpr size_t _NUMBER_OF_CELLS_DEFAULT_     = 7;       // for each dimension

    static constexpr size_t _NOSE_HOOVER_CHAIN_LENGTH_DEFAULT_     = 3;       // default value for nose hoover chain length
    static constexpr double _BERENDSEN_THERMOSTAT_RELAXATION_TIME_ = 0.1;     // in ps
    static constexpr double _LANGEVIN_THERMOSTAT_FRICTION_         = 1.0e11;  // in s^-1
    static constexpr double _NOSE_HOOVER_COUPLING_FREQUENCY_       = 1.0e3;   // in cm^-1

    static constexpr double _BERENDSEN_MANOSTAT_RELAX_TIME_ = 1.0;        // in ps
    static constexpr double _COMPRESSIBILITY_WATER_DEFAULT_ = 4.591e-5;   // default value for berendsen manostat

    static constexpr size_t _DIMENSIONALITY_DEFAULT_ = 3;

    static constexpr double _QM_LOOP_TIME_LIMIT_DEFAULT_ = -1.0;   // in s

    static constexpr char   _OPTIMIZER_DEFAULT_[]           = "gradient-descent";
    static constexpr size_t _N_EPOCHS_DEFAULT_              = 100;
    static constexpr size_t _LR_UPDATE_FREQUENCY_DEFAULT_   = 1;
    static constexpr double _INITIAL_LEARNING_RATE_DEFAULT_ = 1.0;
    static constexpr double _MIN_LEARNING_RATE_DEFAULT_     = 1e-15;

    static constexpr char   _EN_CONV_STRATEGY_DEFAULT_[] = "rigorous";
    static constexpr char   _FORCE_CONV_STRATEGY_DEFAULT_[]  = "rigorous";
    static constexpr double _REL_ENERGY_CONV_DEFAULT_        = 1.0e-6;
    static constexpr double _ABS_ENERGY_CONV_DEFAULT_        = 1.0e-6;
    static constexpr double _MAX_FORCE_CONV_DEFAULT_         = 1.0e-6;
    static constexpr double _RMS_FORCE_CONV_DEFAULT_         = 1.0e-6;

    // clang-format on

}   // namespace defaults

#endif   // _DEFAULTS_HPP_