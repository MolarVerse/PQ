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
    static constexpr double FORCE_UNIT_TO_SI    = KCAL_TO_J / AVOGADRO_NUMBER / ANGSTROM_TO_M;
    static constexpr double MASS_UNIT_TO_SI     = AMU_TO_KG;
    static constexpr double TIME_UNIT_TO_SI     = FS_TO_S;
    static constexpr double VELOCITY_UNIT_TO_SI = ANGSTROM_TO_M;
    static constexpr double ENERGY_UNIT_TO_SI   = KCAL_TO_J / AVOGADRO_NUMBER;
    static constexpr double VOLUME_UNIT_TO_SI   = ANGSTROM3_TO_M3;
    static constexpr double PRESSURE_UNIT_TO_SI = BAR_TO_P;
    static constexpr double LENGTH_UNIT_TO_SI   = ANGSTROM_TO_M;
    static constexpr double MOMENTUM_UNIT_TO_SI = G_TO_KG * ANGSTROM_TO_M / AVOGADRO_NUMBER;
    // clang-format on

    /**
     * @brief Conversion factors to internal units
     */
    static constexpr double SI_TO_VELOCITY_UNIT = 1.0 / VELOCITY_UNIT_TO_SI;
    static constexpr double SI_TO_ENERGY_UNIT   = 1.0 / ENERGY_UNIT_TO_SI;
    static constexpr double SI_TO_PRESSURE_UNIT = 1.0 / PRESSURE_UNIT_TO_SI;
    static constexpr double SI_TO_LENGTH_UNIT   = 1.0 / LENGTH_UNIT_TO_SI;
    static constexpr double SI_TO_FORCE_UNIT    = 1.0 / FORCE_UNIT_TO_SI;

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
    static constexpr double V_VERLET_VELOCITY_FACTOR =
        0.5 * (FORCE_UNIT_TO_SI / MASS_UNIT_TO_SI) * TIME_UNIT_TO_SI *
        SI_TO_VELOCITY_UNIT;

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
    static constexpr double TEMPERATURE_FACTOR =
        VELOCITY_UNIT_TO_SI * VELOCITY_UNIT_TO_SI * MASS_UNIT_TO_SI /
        BOLTZMANN_CONSTANT;

    /**
     * @brief Conversion factors kinetic energy
     *
     * @details E = 0.5 * m v²
     *
     * [E] = kcal mol⁻¹
     * [m] = g mol⁻¹
     * [v] = A s⁻¹
     */
    static constexpr double KINETIC_ENERGY_FACTOR =
        0.5 * MASS_UNIT_TO_SI * VELOCITY_UNIT_TO_SI * VELOCITY_UNIT_TO_SI *
        SI_TO_ENERGY_UNIT;

    /**
     * @brief Conversion factors for pressure calculation
     *
     * @details P = E / V
     *
     * [P] = bar
     * [E] = kcal mol⁻¹
     * [V] = A³
     */
    static constexpr double PRESSURE_FACTOR =
        ENERGY_UNIT_TO_SI / VOLUME_UNIT_TO_SI * SI_TO_PRESSURE_UNIT;

    /**
     * @brief Conversion factors for coulomb preFactor
     *
     * @details factor = 1 / (4 * pi * eps0)
     *          E = factor * q1 * q2 / r
     *
     * [factor] = kcal mol⁻¹ A e²
     * [eps0]   = F m⁻¹
     */
    static constexpr double COULOMB_PREFACTOR =
        1 / (4 * M_PI * PERMITTIVITY_VACUUM) * ELECTRON_CHARGE2 *
        SI_TO_ENERGY_UNIT * SI_TO_LENGTH_UNIT;

    /**
     * @brief ring polymer molecular dynamics
     *
     * @TODO: add details
     */
    static constexpr double RPMD_PREFACTOR =
        BOLTZMANN_CONSTANT2 / REDUCED_PLANCK_CONSTANT2 / M2_TO_ANGSTROM2 *
        G_TO_KG * J_TO_KCAL;

    /**
     * @brief conversion factor for the momentum factor to force * s
     *
     * @details F = p / dt
     *
     * [F] = kcal mol⁻¹ A⁻¹
     * [p] = g mol⁻¹ A s⁻¹
     * [dt] = s
     */
    static constexpr double MOMENTUM_TO_FORCE =
        MOMENTUM_UNIT_TO_SI * SI_TO_FORCE_UNIT;

    static constexpr auto NOSE_HOVER_FRICTION_INPUT_TO_INTERNAL = 1.0e12;

}   // namespace constants

#endif   // _INTERNAL_CONVERSION_FACTORS_HPP_
