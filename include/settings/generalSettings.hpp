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

#ifndef SETTINGS__GENERAL_SETTINGS_HPP
#define SETTINGS__GENERAL_SETTINGS_HPP

#include <cstdint>
#include <mstd/enum.hpp>

#include "defaults.hpp"
#include "enums/general.hpp"
#include "enums/integrator.hpp"
#include "enums/jobtype.hpp"
#include "enums/virial.hpp"
#include "setting.hpp"

/**
 * @class Settings
 *
 * @brief Stores the general settings of the simulation
 *
 */
class GeneralSettings
{
   private:
    Setting<JobType, "JobType"> _jobtype;

    Setting<FPType, "FloatingPointType"> _floatingPointType{FPType::DOUBLE};
    Setting<uint_fast32_t, "RandomSeed"> _randomSeed;

    static inline size_t _dimensionality = defaults::DIMENSIONALITY_DEFAULT;

    static inline VirialType _virial = VirialType::MOLECULAR;

    // setting velocity verlet as default integrator type for backward
    // compatibility until 0.7.0 it was not necessary to specify the
    // integrator type in the input file, so we set it to velocity verlet by
    // default
    static inline IntegratorType _integrator = IntegratorType::VELOCITY_VERLET;

    static inline bool _isCellListActivated =
        defaults::CELL_LIST_IS_ACTIVE_DEFAULT;

    static inline size_t _numberOfCells = defaults::NUMBER_OF_CELLS_DEFAULT;

    static inline bool _isIntraNonBondedActivated = false;

   public:
    /***************************
     * standard setter methods *
     ***************************/

    void setJobtypeByUser(JobType jobtype);

    void setFloatingPointTypeByUser(FPType);
    void setRandomSeedByUser(uint_fast32_t randomSeed);

    static void setDimensionality(size_t dimensionality);

    static void setVirialType(VirialType virialType);
    static void setIntegratorType(IntegratorType integratorType);

    static void setNumberOfCells(size_t numberOfCells);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] JobType getJobtype() const;
    [[nodiscard]] bool    isJobtypeSetByUser() const;

    [[nodiscard]] FPType      getFloatingPointType() const;
    [[nodiscard]] std::string getFloatingPointPybindString() const;

    [[nodiscard]] uint_fast32_t getRandomSeed() const;
    [[nodiscard]] bool          isRandomSeedSetByUser() const;

    [[nodiscard]] static size_t getDimensionality();

    [[nodiscard]] static VirialType     getVirialType();
    [[nodiscard]] static IntegratorType getIntegratorType();
    [[nodiscard]] static size_t         getNumberOfCells();

    /******************************
     * standard is-active methods *
     ******************************/

    static void activateCellList();
    static void deactivateCellList();
    static void activateIntraNonBonded();
    static void deactivateIntraNonBonded();

    [[nodiscard]] static bool isCellListActivated();
    [[nodiscard]] static bool isIntraNonBondedActivated();
};

#endif   // SETTINGS__GENERAL_SETTINGS_HPP
