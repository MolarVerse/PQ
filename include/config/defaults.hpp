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

#include <cstddef>   // for size_t

/**
 * @brief struct containing all default file names
 *
 */
struct DefaultFiles
{
    static constexpr auto restartFile  = "default.rst";
    static constexpr auto energyFile   = "default.en";
    static constexpr auto instEnFile   = "default.instant_en";
    static constexpr auto momentumFile = "default.mom";
    static constexpr auto trajFile     = "default.xyz";
    static constexpr auto velFile      = "default.vel";
    static constexpr auto forceFile    = "default.force";
    static constexpr auto chargeFile   = "default.charge";
    static constexpr auto logFile      = "default.log";
    static constexpr auto stdoutFile   = "default.stdout";
    static constexpr auto refFile      = "default.ref";
    static constexpr auto infoFile     = "default.info";
    static constexpr auto virialFile   = "default.vir";
    static constexpr auto stressFile   = "default.stress";
    static constexpr auto boxFile      = "default.box";
    static constexpr auto optFile      = "default.opt";
    static constexpr auto timingsFile  = "default.timings";

    static constexpr auto rpmdRstFile    = "default.rpmd.rst";
    static constexpr auto rpmdTrajFile   = "default.rpmd.xyz";
    static constexpr auto rpmdVelFile    = "default.rpmd.vel";
    static constexpr auto rpmdForceFile  = "default.rpmd.force";
    static constexpr auto rpmdChargeFile = "default.rpmd.charge";
    static constexpr auto rpmdEnergyFile = "default.rpmd.en";
};

/**
 * @brief namespace containing all default values
 *
 */
namespace defaults
{

    // clang-format off
    static constexpr char   _MOLDESCRIPTOR_FILE_DEFAULT_[] = "moldescriptor.dat";
    static constexpr char   _GUFF_FILE_DEFAULT_[]          = "guff.dat";
    static constexpr char   _DFTB_FILE_DEFAULT_[]          = "dftb_in.template";
    static constexpr size_t _NUMBER_OF_GUFF_ENTRIES_       = 28;

    static constexpr char   _HESSIAN_FILE_DEFAULT_[] = "default.hessian";
    static constexpr char   _HESSIAN_INFO_FILE_DEFAULT_[] =
        "default.hessian.info";
    static constexpr double _HESSIAN_DISPLACEMENT_DEFAULT_ = 1.0e-3;
    static constexpr bool   _HESSIAN_OPTIMIZE_DEFAULT_     = true;
    static constexpr char   _HESSIAN_BUILDER_DEFAULT_[]    = "central";

    static constexpr char _QM_FORCES_TEMP_FILE_DEFAULT_[]     = "qm_forces";
    static constexpr char _QM_CHARGES_TEMP_FILE_DEFAULT_[]    = "qm_charges";
    static constexpr char _STRESS_TENSOR_TEMP_FILE_DEFAULT_[] = "stress_tensor";

    static constexpr double _COULOMB_CUT_OFF_DEFAULT_           = 12.5;   // in Angstrom
    static constexpr double _SCALE_14_COULOMB_DEFAULT_          = 1.0;
    static constexpr double _SCALE_14_VAN_DER_WAALS_DEFAULT_    = 1.0;
    static constexpr double _WOLF_PARAM_DEFAULT_            = 0.25;     // TODO: add unit

    static constexpr bool   _CONSTRAINTS_ACTIVE_DEFAULT_ = false;
    static constexpr size_t _SHAKE_MAX_ITER_DEFAULT_     = 20;
    static constexpr size_t _RATTLE_MAX_ITER_DEFAULT_    = 20;
    static constexpr size_t _MSHAKE_MAX_ITER_DEFAULT_    = 20;
    static constexpr double _SHAKE_TOLERANCE_DEFAULT_    = 1e-8;
    static constexpr double _RATTLE_TOLERANCE_DEFAULT_   = 1e-8 * 1e12;
    static constexpr double _MSHAKE_TOLERANCE_DEFAULT_   = 1e-8;

    static constexpr bool   _CELL_LIST_IS_ACTIVE_DEFAULT_ = false;   // default is brute force routine
    static constexpr size_t _NUMBER_OF_CELLS_DEFAULT_     = 7;       // for each dimension

    static constexpr size_t _NH_CHAIN_LENGTH_DEFAULT_     = 3;       // default value for nose hoover chain length
    static constexpr double _BERENDSEN_THERMOSTAT_RELAX_TIME_ = 0.1;     // in ps
    static constexpr double _LANGEVIN_THERMOSTAT_FRICTION_         = 1.0e11;  // in s^-1
    static constexpr double _NH_COUPLING_FREQ_       = 1.0e3;   // in cm^-1

    static constexpr double _BERENDSEN_MANOSTAT_RELAX_TIME_ = 1.0;        // in ps
    static constexpr double _COMPRESSIBILITY_WATER_DEFAULT_ = 4.591e-5;   // in bar^-1 default value for berendsen manostat

    static constexpr size_t _DIMENSIONALITY_DEFAULT_ = 3;

    static constexpr double _QM_LOOP_TIME_LIMIT_DEFAULT_ = 3600;   // in s

    static constexpr char   _OPTIMIZER_DEFAULT_[]           = "gradient-descent";
    static constexpr size_t _N_EPOCHS_DEFAULT_              = 100;
    static constexpr size_t _LR_UPDATE_FREQUENCY_DEFAULT_   = 1;
    static constexpr double _INITIAL_LEARNING_RATE_DEFAULT_ = 1.0e-4;
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
