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

#ifndef _CONVERSION_FACTORS_HPP_

#define _CONVERSION_FACTORS_HPP_

#include <cmath>

#include "natureConstants.hpp"

#ifndef M_PI
#define M_PI std::numbers::pi
#endif

namespace constants
{
    /**
     * @brief conversion factors for degrees
     *
     */
    static constexpr double DEG_TO_RAD = M_PI / 180.0;
    static constexpr double RAD_TO_DEG = 180.0 / M_PI;

    /**
     * @brief Conversion factors for mass units
     */
    static constexpr double G_TO_KG    = 1.0e-3;
    static constexpr double KG_TO_GRAM = 1.0 / G_TO_KG;
    static constexpr double AMU_TO_KG  = G_TO_KG / AVOGADRO_NUMBER;
    static constexpr double KG_TO_AMU  = 1.0 / AMU_TO_KG;

    /**
     * @brief Conversion factors for length units
     */
    static constexpr double ANGSTROM_TO_M = 1.0e-10;
    static constexpr double M_TO_ANGSTROM = 1.0 / ANGSTROM_TO_M;

    static constexpr double BOHR_TO_M = BOHR_RADIUS;
    static constexpr double M_TO_BOHR = 1.0 / BOHR_TO_M;

    static constexpr double ANGSTROM_TO_BOHR = ANGSTROM_TO_M / BOHR_TO_M;
    static constexpr double BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR;

    /**
     * @brief Conversion factors for area units
     */
    // clang-format off
    static constexpr double ANGSTROM2_TO_M2 = ANGSTROM_TO_M * ANGSTROM_TO_M;
    static constexpr double M2_TO_ANGSTROM2 = 1 / ANGSTROM2_TO_M2;
    // clang-format on

    /**
     * @brief Conversion factors for volume units
     */
    // clang-format off
    static constexpr double ANGSTROM3_TO_M3     = ANGSTROM_TO_M * ANGSTROM_TO_M * ANGSTROM_TO_M;
    static constexpr double M3_TO_ANGSTROM3    = 1.0 / ANGSTROM3_TO_M3;
    static constexpr double ANGSTROM3_TO_L     = ANGSTROM3_TO_M3 * 1.0e3;
    static constexpr double L_TO_ANGSTROM3     = 1.0 / ANGSTROM3_TO_L;
    static constexpr double BOHR3_TO_ANGSTROM3 = BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM;
    // clang-format on

    /**
     * @brief Conversion factors for density units
     */
    // clang-format off
    static constexpr double KG_PER_L_TO_AMU_PER_ANGSTROM3 = KG_TO_AMU / L_TO_ANGSTROM3;
    static constexpr double AMU_PER_ANGSTROM3_TO_KG_PER_L = 1.0 / KG_PER_L_TO_AMU_PER_ANGSTROM3;
    // clang-format on

    /**
     * @brief Conversion factors for energy units
     */
    // clang-format off
    static constexpr double KCAL_TO_J                          = 4184.0;
    static constexpr double J_TO_KCAL                          = 1.0 / KCAL_TO_J;
    static constexpr double J_TO_KCAL_PER_MOL                  = J_TO_KCAL * AVOGADRO_NUMBER;
    static constexpr double KCAL_PER_MOL_TO_J                  = 1.0 / J_TO_KCAL_PER_MOL;
    static constexpr double EV_TO_J                            = 1.602176634e-19;
    static constexpr double EV_TO_KCAL_PER_MOL                 = EV_TO_J * J_TO_KCAL_PER_MOL;
    static constexpr double HARTREE_TO_KCAL_PER_MOL            = 627.5096080305927;
    static constexpr double BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL = BOLTZMANN_CONSTANT * J_TO_KCAL_PER_MOL;
    // clang-format on

    /**
     * @brief Conversion factors for squared energy units
     */
    // clang-format off
    static constexpr double BOLTZMANN_CONSTANT2      = BOLTZMANN_CONSTANT * BOLTZMANN_CONSTANT;
    static constexpr double REDUCED_PLANCK_CONSTANT2 = REDUCED_PLANCK_CONSTANT * REDUCED_PLANCK_CONSTANT;
    // clang-format on

    /**
     * @brief Conversion factors for force units
     */
    static constexpr double HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM =
        HARTREE_TO_KCAL_PER_MOL / BOHR_TO_ANGSTROM;

    /**
     * @brief Conversion factors for stress units
     */
    static constexpr double HARTREE_PER_BOHR3_TO_KCAL_PER_MOL_PER_ANGSTROM3 =
        HARTREE_TO_KCAL_PER_MOL / BOHR3_TO_ANGSTROM3;

    /**
     * @brief Conversion factors for time units
     */
    static constexpr double S_TO_FS  = 1.0e15;
    static constexpr double FS_TO_S  = 1.0 / S_TO_FS;
    static constexpr double PS_TO_FS = 1.0e3;
    static constexpr double FS_TO_PS = 1.0 / PS_TO_FS;

    /**
     * @brief Conversion factors for pressure calculation
     */
    static constexpr double P_TO_BAR = 1.0e-5;
    static constexpr double BAR_TO_P = 1.0 / P_TO_BAR;

    /**
     * @brief Conversion factors for velocities
     */
    static constexpr double M_PER_S_TO_CM_PER_S = 1.0e2;
    static constexpr double SPEED_OF_LIGHT_IN_CM_PER_S =
        SPEED_OF_LIGHT * M_PER_S_TO_CM_PER_S;

    /**
     * @brief Conversion factors for frequencies
     */
    static constexpr double PER_CM_TO_HZ = SPEED_OF_LIGHT_IN_CM_PER_S;

}   // namespace constants

#endif   // _CONVERSION_FACTORS_HPP_
