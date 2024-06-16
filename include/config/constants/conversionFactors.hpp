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
    static constexpr double _DEG_TO_RAD_ = M_PI / 180.0;
    static constexpr double _RAD_TO_DEG_ = 180.0 / M_PI;

    /**
     * @brief Conversion factors for mass units
     */
    static constexpr double _G_TO_KG_    = 1.0e-3;
    static constexpr double _KG_TO_GRAM_ = 1.0 / _G_TO_KG_;
    static constexpr double _AMU_TO_KG_  = _G_TO_KG_ / _AVOGADRO_NUMBER_;
    static constexpr double _KG_TO_AMU_  = 1.0 / _AMU_TO_KG_;

    /**
     * @brief Conversion factors for length units
     */
    static constexpr double _ANGSTROM_TO_M_ = 1.0e-10;
    static constexpr double _M_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_M_;

    static constexpr double _BOHR_TO_M_ = _BOHR_RADIUS_;
    static constexpr double _M_TO_BOHR_ = 1.0 / _BOHR_TO_M_;

    static constexpr double _ANGSTROM_TO_BOHR_ = _ANGSTROM_TO_M_ / _BOHR_TO_M_;
    static constexpr double _BOHR_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_BOHR_;

    /**
     * @brief Conversion factors for area units
     */
    // clang-format off
    static constexpr double _ANGSTROM2_TO_M2_ = _ANGSTROM_TO_M_ * _ANGSTROM_TO_M_;
    static constexpr double _M2_TO_ANGSTROM2_ = 1 / _ANGSTROM2_TO_M2_;
    // clang-format on

    /**
     * @brief Conversion factors for volume units
     */
    // clang-format off
    static constexpr double _ANGSTROM3_TO_M3     = _ANGSTROM2_TO_M2_ * _ANGSTROM_TO_M_ * _ANGSTROM_TO_M_;
    static constexpr double _M3_TO_ANGSTROM3_    = 1.0 / _ANGSTROM3_TO_M3;
    static constexpr double _ANGSTROM3_TO_L_     = _ANGSTROM3_TO_M3 * 1.0e3;
    static constexpr double _L_TO_ANGSTROM3_     = 1.0 / _ANGSTROM3_TO_L_;
    static constexpr double _BOHR3_TO_ANGSTROM3_ = _BOHR_TO_ANGSTROM_ * _BOHR_TO_ANGSTROM_ * _BOHR_TO_ANGSTROM_;
    // clang-format on

    /**
     * @brief Conversion factors for density units
     */
    // clang-format off
    static constexpr double _KG_PER_L_TO_AMU_PER_ANGSTROM3_ = _KG_TO_AMU_ / _L_TO_ANGSTROM3_;
    static constexpr double _AMU_PER_ANGSTROM3_TO_KG_PER_L_ = 1.0 / _KG_PER_L_TO_AMU_PER_ANGSTROM3_;
    // clang-format on

    /**
     * @brief Conversion factors for energy units
     */
    // clang-format off
    static constexpr double _KCAL_TO_J_                          = 4184.0;
    static constexpr double _J_TO_KCAL_                          = 1.0 / _KCAL_TO_J_;
    static constexpr double _J_TO_KCAL_PER_MOL_                  = _J_TO_KCAL_ * _AVOGADRO_NUMBER_;
    static constexpr double _KCAL_PER_MOL_TO_J_                  = 1.0 / _J_TO_KCAL_PER_MOL_;
    static constexpr double _HARTREE_TO_KCAL_PER_MOL_            = 627.5096080305927;
    static constexpr double _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ = _BOLTZMANN_CONSTANT_ * _J_TO_KCAL_PER_MOL_;
    // clang-format on

    /**
     * @brief Conversion factors for squared energy units
     */
    // clang-format off
    static constexpr double _BOLTZMANN_CONSTANT2_      = _BOLTZMANN_CONSTANT_ * _BOLTZMANN_CONSTANT_;
    static constexpr double _REDUCED_PLANCK_CONSTANT2_ = _REDUCED_PLANCK_CONSTANT_ * _REDUCED_PLANCK_CONSTANT_;
    // clang-format on

    /**
     * @brief Conversion factors for force units
     */
    static constexpr double _HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_ =
        _HARTREE_TO_KCAL_PER_MOL_ / _BOHR_TO_ANGSTROM_;

    /**
     * @brief Conversion factors for stress units
     */
    static constexpr double _HARTREE_PER_BOHR3_TO_KCAL_PER_MOL_PER_ANGSTROM3_ =
        _HARTREE_TO_KCAL_PER_MOL_ / _BOHR3_TO_ANGSTROM3_;

    /**
     * @brief Conversion factors for time units
     */
    static constexpr double _S_TO_FS_  = 1.0e15;
    static constexpr double _FS_TO_S_  = 1.0 / _S_TO_FS_;
    static constexpr double _PS_TO_FS_ = 1.0e3;
    static constexpr double _FS_TO_PS_ = 1.0 / _PS_TO_FS_;

    /**
     * @brief Conversion factors for pressure calculation
     */
    static constexpr double _P_TO_BAR_ = 1.0e-5;
    static constexpr double _BAR_TO_P_ = 1.0 / _P_TO_BAR_;

    /**
     * @brief Conversion factors for velocities
     */
    static constexpr double _M_PER_S_TO_CM_PER_S_ = 1.0e2;
    static constexpr double _SPEED_OF_LIGHT_IN_CM_PER_S_ =
        _SPEED_OF_LIGHT_ * _M_PER_S_TO_CM_PER_S_;

    /**
     * @brief Conversion factors for frequencies
     */
    static constexpr double _PER_CM_TO_HZ_ = _SPEED_OF_LIGHT_IN_CM_PER_S_;

}   // namespace constants

#endif   // _CONVERSION_FACTORS_HPP_