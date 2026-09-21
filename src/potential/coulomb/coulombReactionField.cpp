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

#include "coulombReactionField.hpp"

#include "constants/internalConversionFactors.hpp"

using namespace pot;

/**
 * @brief Construct a new Coulomb Reaction Field object
 *
 * @param coulombRadiusCutOff
 * @param epsilon
 */
CoulombReactionField::CoulombReactionField(
    double coulombRadiusCutOff,
    double epsilon
)
    : CoulombPotential{coulombRadiusCutOff},
      _rfPrefactor{(epsilon - 1) / (2.0 * epsilon + 1)}
{
}

/**
 * @brief calculate the energy and force of the reaction field Coulomb potential
 *
 * @param distance
 * @param chargeProduct
 * @return std::pair<double, double>
 */
std::pair<double, double> CoulombReactionField::calculate(
    double distance,
    double chargeProduct
) const
{
    const auto coulombPrefactor = chargeProduct * COULOMB_PREFACTOR;
    const auto dInv             = 1.0 / distance;
    const auto deltaCutOff      = _coulombRadiusCutOff - distance;
    const auto rCutEnergy       = _coulombEnergyCutOff;
    const auto rCutForce        = _coulombForceCutOff;
    const auto rfCutOffCubed    = _rfPrefactor * _coulombCutOffCubedInverse;

    auto energy  = dInv - (2.0 * rCutEnergy) + (distance * rCutForce);
    energy      += rfCutOffCubed * deltaCutOff * deltaCutOff;

    auto force  = (dInv * dInv) - rCutForce;
    force      += 2.0 * rfCutOffCubed * deltaCutOff;

    energy *= coulombPrefactor;
    force  *= coulombPrefactor;

    return {energy, force};
}
