/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "coulombShiftedPotential.hpp"

#include "constants/internalConversionFactors.hpp"   // for _COULOMB_PREFACTOR_

using namespace potential;

/**
 * @brief calculate the energy and force of the shifted Coulomb potential
 *
 * @param distance
 * @return std::pair<double, double>
 */
[[nodiscard]] std::pair<double, double> CoulombShiftedPotential::calculate(const double distance,
                                                                           const double chargeProduct) const
{
    const auto coulombPrefactor = chargeProduct * constants::_COULOMB_PREFACTOR_;

    auto energy = (1 / distance) - _coulombEnergyCutOff - _coulombForceCutOff * (_coulombRadiusCutOff - distance);
    auto force  = (1 / (distance * distance)) - _coulombForceCutOff;

    energy *= coulombPrefactor;
    force  *= coulombPrefactor;

    return {energy, force};
}