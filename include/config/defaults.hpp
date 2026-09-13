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
    static constexpr auto restartFile      = "default.rst";
    static constexpr auto energyFile       = "default.en";
    static constexpr auto instEnFile       = "default.instant_en";
    static constexpr auto momentumFile     = "default.mom";
    static constexpr auto trajFile         = "default.xyz";
    static constexpr auto hybridCenterFile = "default.center.xyz";
    static constexpr auto velFile          = "default.vel";
    static constexpr auto forceFile        = "default.force";
    static constexpr auto chargeFile       = "default.charge";
    static constexpr auto logFile          = "default.log";
    static constexpr auto stdoutFile       = "default.stdout";
    static constexpr auto refFile          = "default.ref";
    static constexpr auto infoFile         = "default.info";
    static constexpr auto virialFile       = "default.vir";
    static constexpr auto stressFile       = "default.stress";
    static constexpr auto boxFile          = "default.box";
    static constexpr auto optFile          = "default.opt";
    static constexpr auto timingsFile      = "default.timings";

    static constexpr auto rpmdRstFile    = "default.rpmd.rst";
    static constexpr auto rpmdTrajFile   = "default.rpmd.xyz";
    static constexpr auto rpmdVelFile    = "default.rpmd.vel";
    static constexpr auto rpmdForceFile  = "default.rpmd.force";
    static constexpr auto rpmdChargeFile = "default.rpmd.charge";
    static constexpr auto rpmdEnergyFile = "default.rpmd.en";

    static constexpr auto hessianFile     = "default.hessian";
    static constexpr auto hessianInfoFile = "default.hessian.info";
};

/**
 * @brief namespace containing all default values
 *
 */
namespace defaults
{

    static constexpr auto*  MOLDESCRIPTOR_FILE_DEFAULT = "moldescriptor.dat";
    static constexpr auto*  GUFF_FILE_DEFAULT          = "guff.dat";
    static constexpr auto*  DFTB_FILE_DEFAULT          = "dftb_in.template";
    static constexpr auto*  TM_FILE_DEFAULT            = "tm_define.template";
    static constexpr auto*  POINTCHARGE_FILE_DEFAULT   = "pointcharges";
    static constexpr size_t NUMBER_OF_GUFF_ENTRIES     = 28;

    static constexpr auto* QM_FORCES_TEMP_FILE_DEFAULT     = "qm_forces";
    static constexpr auto* QM_CHARGES_TEMP_FILE_DEFAULT    = "qm_charges";
    static constexpr auto* STRESS_TENSOR_TEMP_FILE_DEFAULT = "stress_tensor";

    static constexpr double HESSIAN_DISPLACEMENT_DEFAULT = 1.0e-3;
    static constexpr bool   HESSIAN_OPTIMIZE_DEFAULT     = true;
    static constexpr auto*  HESSIAN_BUILDER_DEFAULT      = "central";

    static constexpr char INNER_REGION_CENTER_ATOM_NAME = 'X';

    static constexpr double COULOMB_CUT_OFF_DEFAULT  = 12.5;   // in Angstrom
    static constexpr double SCALE_14_COULOMB_DEFAULT = 1.0;
    static constexpr double SCALE_14_VAN_DER_WAALS_DEFAULT = 1.0;
    static constexpr double WOLF_PARAM_DEFAULT = 0.25;   // TODO: add unit

    static constexpr bool   CONSTRAINTS_ACTIVE_DEFAULT = false;
    static constexpr size_t SHAKE_MAX_ITER_DEFAULT     = 20;
    static constexpr size_t RATTLE_MAX_ITER_DEFAULT    = 20;
    static constexpr size_t MSHAKE_MAX_ITER_DEFAULT    = 20;
    static constexpr double SHAKE_TOLERANCE_DEFAULT    = 1e-8;
    static constexpr double RATTLE_TOLERANCE_DEFAULT   = 1e-8 * 1e12;
    static constexpr double MSHAKE_TOLERANCE_DEFAULT   = 1e-8;

    static constexpr bool CELL_LIST_IS_ACTIVE_DEFAULT =
        false;   // default is brute force routine
    static constexpr size_t NUMBER_OF_CELLS_DEFAULT = 7;   // for each dimension

    static constexpr size_t NH_CHAIN_LENGTH_DEFAULT =
        3;   // default value for nose hoover chain length
    static constexpr double BERENDSEN_THERMOSTAT_RELAX_TIME = 0.1;   // in ps
    static constexpr double LANGEVIN_THERMOSTAT_FRICTION = 1.0e11;   // in s^-1
    static constexpr double NH_COUPLING_FREQ             = 1.0e3;    // in cm^-1
    static constexpr double MAX_FRICTION_CONVERSION      = 1.0e12;   // in s^-1

    static constexpr double BERENDSEN_MANOSTAT_RELAX_TIME = 1.0;   // in ps
    static constexpr double COMPRESSIBILITY_WATER_DEFAULT =
        4.591e-5;   // in bar^-1 default value for berendsen manostat

    static constexpr size_t DIMENSIONALITY_DEFAULT = 3;

    static constexpr double QM_LOOP_TIME_LIMIT_DEFAULT = 3600;   // in s
    static constexpr double VACUUM_BOX_DIMENSION       = 1000;   // in Å

    static constexpr auto*  OPTIMIZER_DEFAULT             = "gradient-descent";
    static constexpr size_t N_EPOCHS_DEFAULT              = 100;
    static constexpr size_t LR_UPDATE_FREQUENCY_DEFAULT   = 1;
    static constexpr double INITIAL_LEARNING_RATE_DEFAULT = 1.0e-4;
    static constexpr double MIN_LEARNING_RATE_DEFAULT     = 1e-15;

    static constexpr auto*  EN_CONV_STRATEGY_DEFAULT    = "rigorous";
    static constexpr auto*  FORCE_CONV_STRATEGY_DEFAULT = "rigorous";
    static constexpr double REL_ENERGY_CONV_DEFAULT     = 1.0e-6;
    static constexpr double ABS_ENERGY_CONV_DEFAULT     = 1.0e-6;
    static constexpr double MAX_FORCE_CONV_DEFAULT      = 1.0e-6;
    static constexpr double RMS_FORCE_CONV_DEFAULT      = 1.0e-6;

    static constexpr auto NUM_GUFF_COEFFICIENTS = 22;
    // clang-format on

}   // namespace defaults

#endif   // _DEFAULTS_HPP_
