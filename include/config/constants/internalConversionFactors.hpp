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

#ifndef _INTERNAL_CONVERSION_FACTORS_HPP_

#define _INTERNAL_CONVERSION_FACTORS_HPP_

#include "conversionFactors.hpp"
#include "natureConstants.hpp"

#ifndef M_PI
#define M_PI std::numbers::pi
#endif

namespace constants
{
    /**
     * @brief Conversion factors to SI units
     */
    // clang-format off
    static constexpr double _FORCE_UNIT_TO_SI_    = _KCAL_TO_J_ / _AVOGADRO_NUMBER_ / _ANGSTROM_TO_M_;
    static constexpr double _MASS_UNIT_TO_SI_     = _AMU_TO_KG_;
    static constexpr double _TIME_UNIT_TO_SI_     = _FS_TO_S_;
    static constexpr double _VELOCITY_UNIT_TO_SI_ = _ANGSTROM_TO_M_;
    static constexpr double _ENERGY_UNIT_TO_SI_   = _KCAL_TO_J_ / _AVOGADRO_NUMBER_;
    static constexpr double _VOLUME_UNIT_TO_SI_   = _ANGSTROM3_TO_M3;
    static constexpr double _PRESSURE_UNIT_TO_SI_ = _BAR_TO_P_;
    static constexpr double _LENGTH_UNIT_TO_SI_   = _ANGSTROM_TO_M_;
    static constexpr double _MOMENTUM_UNIT_TO_SI_ = _G_TO_KG_ * _ANGSTROM_TO_M_ / _AVOGADRO_NUMBER_;
    // clang-format on

    /**
     * @brief Conversion factors to internal units
     */
    static constexpr double _SI_TO_VELOCITY_UNIT_ = 1.0 / _VELOCITY_UNIT_TO_SI_;
    static constexpr double _SI_TO_ENERGY_UNIT_   = 1.0 / _ENERGY_UNIT_TO_SI_;
    static constexpr double _SI_TO_PRESSURE_UNIT_ = 1.0 / _PRESSURE_UNIT_TO_SI_;
    static constexpr double _SI_TO_LENGTH_UNIT_   = 1.0 / _LENGTH_UNIT_TO_SI_;
    static constexpr double _SI_TO_FORCE_UNIT_    = 1.0 / _FORCE_UNIT_TO_SI_;

    /**
     * @brief Conversion factor for velocity verlet integrator
     *
     * @details v = 0.5 * F * dt / m
     *
     * [v] = A s⁻¹
     * [F] = kcal mol⁻¹ A⁻¹
     * [dt] = fs
     * [m] = g mol⁻¹
     */
    static constexpr double _V_VERLET_VELOCITY_FACTOR_ =
        0.5 * (_FORCE_UNIT_TO_SI_ / _MASS_UNIT_TO_SI_) * _TIME_UNIT_TO_SI_ *
        _SI_TO_VELOCITY_UNIT_;

    /**
     * @brief Conversion factors for temperature calculation
     *
     * @details T = m v² / kB
     *
     * [T]  = K
     * [m]  = g mol⁻¹
     * [v]  = A s⁻¹
     * [kb] = J K⁻¹
     */
    static constexpr double _TEMPERATURE_FACTOR_ =
        _VELOCITY_UNIT_TO_SI_ * _VELOCITY_UNIT_TO_SI_ * _MASS_UNIT_TO_SI_ /
        _BOLTZMANN_CONSTANT_;

    /**
     * @brief Conversion factors kinetic energy
     *
     * @details E = 0.5 * m v²
     *
     * [E] = kcal mol⁻¹
     * [m] = g mol⁻¹
     * [v] = A s⁻¹
     */
    static constexpr double _KINETIC_ENERGY_FACTOR_ =
        0.5 * _MASS_UNIT_TO_SI_ * _VELOCITY_UNIT_TO_SI_ *
        _VELOCITY_UNIT_TO_SI_ * _SI_TO_ENERGY_UNIT_;

    /**
     * @brief Conversion factors for pressure calculation
     *
     * @details P = E / V
     *
     * [P] = bar
     * [E] = kcal mol⁻¹
     * [V] = A³
     */
    static constexpr double _PRESSURE_FACTOR_ =
        _ENERGY_UNIT_TO_SI_ / _VOLUME_UNIT_TO_SI_ * _SI_TO_PRESSURE_UNIT_;

    /**
     * @brief Conversion factors for coulomb preFactor
     *
     * @details factor = 1 / (4 * pi * eps0)
     *          E = factor * q1 * q2 / r
     *
     * [factor] = kcal mol⁻¹ A e²
     * [eps0]   = F m⁻¹
     */
    static constexpr double _COULOMB_PREFACTOR_ =
        1 / (4 * M_PI * _PERMITTIVITY_VACUUM_) * _ELECTRON_CHARGE2_ *
        _SI_TO_ENERGY_UNIT_ * _SI_TO_LENGTH_UNIT_;

    /**
     * @brief ring polymer molecular dynamics
     *
     * @TODO: add details
     */
    static constexpr double _RPMD_PREFACTOR_ =
        _BOLTZMANN_CONSTANT2_ / _REDUCED_PLANCK_CONSTANT2_ / _M2_TO_ANGSTROM2_ *
        _G_TO_KG_ * _J_TO_KCAL_;

    /**
     * @brief conversion factor for the momentum factor to force * s
     *
     * @details F = p / dt
     *
     * [F] = kcal mol⁻¹ A⁻¹
     * [p] = g mol⁻¹ A s⁻¹
     * [dt] = s
     */
    static constexpr double _MOMENTUM_TO_FORCE_ =
        _MOMENTUM_UNIT_TO_SI_ * _SI_TO_FORCE_UNIT_;

}   // namespace constants

#endif   // _INTERNAL_CONVERSION_FACTORS_HPP_