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

#ifndef _NATURE_CONSTANTS_HPP_

#define _NATURE_CONSTANTS_HPP_

#include <cmath>

#ifndef M_PI
#define M_PI std::numbers::pi
#endif

namespace constants
{
    /**
     * @brief avogadro number in mol⁻¹
     */
    static constexpr double AVOGADRO_NUMBER = 6.02214076e23;

    /**
     * @brief bohr radius in m
     */
    static constexpr double BOHR_RADIUS = 5.2917721067e-11;

    /**
     * @brief Planck constant in J s
     */
    static constexpr double PLANCK_CONSTANT = 6.62607015e-34;
    static constexpr double REDUCED_PLANCK_CONSTANT =
        PLANCK_CONSTANT / (2.0 * M_PI);

    /**
     * @brief Boltzmann constant in J K⁻¹
     * @brief universal gas constant in J mol⁻¹ K⁻¹
     */
    static constexpr double BOLTZMANN_CONSTANT = 1.380649e-23;
    static constexpr double UNIVERSAL_GAS_CONSTANT =
        BOLTZMANN_CONSTANT * AVOGADRO_NUMBER;

    /**
     * @brief electron charge in C
     */
    static constexpr double ELECTRON_CHARGE = 1.602176634e-19;

    /**
     * @brief electron charge squared in C²
     */
    static constexpr double ELECTRON_CHARGE2 =
        ELECTRON_CHARGE * ELECTRON_CHARGE;

    /**
     * @brief electron mass in kg
     */
    static constexpr double ELECTRON_MASS = 9.109389754e-31;

    /**
     * @brief permittivity of vacuum in F/m
     */
    static constexpr double PERMITTIVITY_VACUUM = 8.854187817e-12;

    /**
     * @brief speed of light in m/s
     */
    static constexpr double SPEED_OF_LIGHT = 299792458.0;

}   // namespace constants

#endif   // _NATURE_CONSTANTS_HPP_
