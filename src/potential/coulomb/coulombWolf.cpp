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

#include "coulombWolf.hpp"

#include "constants/internalConversionFactors.hpp"   // for _COULOMB_PREFACTOR_

#include <cmath>   // for exp, sqrt, erfc

using namespace potential;

/**
 * @brief Construct a new Coulomb Wolf:: Coulomb Wolf object
 *
 * @details this constructor calculates automatically the three need wolf parameters from kappa in order to gain speed
 *
 * @param coulombRadiusCutOff
 * @param kappa
 */
CoulombWolf::CoulombWolf(const double coulombRadiusCutOff, const double kappa) : CoulombPotential(coulombRadiusCutOff)
{
    _kappa          = kappa;
    _wolfParameter1 = ::erfc(_kappa * coulombRadiusCutOff) / coulombRadiusCutOff;
    _wolfParameter2 = 2.0 * _kappa / ::sqrt(M_PI);
    _wolfParameter3 = _wolfParameter1 / coulombRadiusCutOff +
                      _wolfParameter2 * ::exp(-_kappa * _kappa * coulombRadiusCutOff * coulombRadiusCutOff) / coulombRadiusCutOff;
}

/**
 * @brief calculate the energy and force of the Coulomb potential with Wolf summation as long range correction
 *
 * @link https://doi.org/10.1063/1.478738
 *
 * @param distance
 * @return std::pair<double, double>
 */
[[nodiscard]] std::pair<double, double> CoulombWolf::calculate(const double distance, const double chargeProduct) const
{

    const auto coulombPrefactor = chargeProduct * constants::_COULOMB_PREFACTOR_;

    const auto kappaDistance = _kappa * distance;
    const auto erfcFactor    = ::erfc(kappaDistance);

    auto energy = erfcFactor / distance - _wolfParameter1 + _wolfParameter3 * (distance - _coulombRadiusCutOff);
    auto force =
        erfcFactor / (distance * distance) + _wolfParameter2 * ::exp(-kappaDistance * kappaDistance) / distance - _wolfParameter3;

    energy *= coulombPrefactor;
    force  *= coulombPrefactor;

    return {energy, force};
}