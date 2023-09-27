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

#include "bondConstraint.hpp"

#include "molecule.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "vector3d.hpp"

#include <cmath>
#include <vector>

using namespace constraints;

/**
 * @brief calculates the reference bond data of a bond constraint
 *
 * @param simulationBox
 */
void BondConstraint::calculateConstraintBondRef(const simulationBox::SimulationBox &simBox)
{
    _shakeDistanceRef = _molecules[0]->getAtomPosition(_atomIndices[0]) - _molecules[1]->getAtomPosition(_atomIndices[1]);

    simBox.applyPBC(_shakeDistanceRef);
}

/**
 * @brief calculates the distance delta of a bond constraint
 *
 */
double BondConstraint::calculateDistanceDelta(const simulationBox::SimulationBox &simBox) const
{
    const auto pos1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto pos2 = _molecules[1]->getAtomPosition(_atomIndices[1]);

    auto dPosition = pos1 - pos2;
    simBox.applyPBC(dPosition);

    const auto distanceSquared = normSquared(dPosition);
    const auto delta           = 0.5 * (_targetBondLength * _targetBondLength - distanceSquared);

    return delta;
}

/**
 * @brief applies the shake algorithm to a bond constraint
 *
 * @details if delta is not smaller than tolerance, the shake algorithm is applied
 *
 */
bool BondConstraint::applyShake(const simulationBox::SimulationBox &simBox, double tolerance)
{

    if (const auto delta = calculateDistanceDelta(simBox); std::fabs(delta / (_targetBondLength * _targetBondLength)) > tolerance)
    {
        const auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
        const auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

        const auto shakeForce = delta / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

        const auto dPosition = shakeForce * _shakeDistanceRef;

        _molecules[0]->addAtomPosition(_atomIndices[0], +invMass1 * dPosition);
        _molecules[1]->addAtomPosition(_atomIndices[1], -invMass2 * dPosition);

        const auto dVelocity = dPosition / settings::TimingsSettings::getTimeStep();

        _molecules[0]->addAtomVelocity(_atomIndices[0], +invMass1 * dVelocity);
        _molecules[1]->addAtomVelocity(_atomIndices[1], -invMass2 * dVelocity);

        return false;
    }

    return true;
}

/**
 * @brief calculates the velocity delta of a bond constraint
 *
 */
[[nodiscard]] double BondConstraint::calculateVelocityDelta() const
{
    const auto dVelocity = _molecules[0]->getAtomVelocity(_atomIndices[0]) - _molecules[1]->getAtomVelocity(_atomIndices[1]);

    const auto scalarProduct = dot(dVelocity, _shakeDistanceRef);

    const auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
    const auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

    const auto delta = -scalarProduct / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

    return delta;
}

/**
 * @brief applies the rattle algorithm to a bond constraint
 *
 * @details if delta is not smaller than tolerance, the rattle algorithm is applied
 *
 */
bool BondConstraint::applyRattle(double tolerance)
{
    if (const auto delta = calculateVelocityDelta(); std::fabs(delta) > tolerance)
    {
        const auto dVelocity = delta * _shakeDistanceRef;

        const auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
        const auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

        _molecules[0]->addAtomVelocity(_atomIndices[0], +invMass1 * dVelocity);
        _molecules[1]->addAtomVelocity(_atomIndices[1], -invMass2 * dVelocity);

        return false;
    }

    return true;
}