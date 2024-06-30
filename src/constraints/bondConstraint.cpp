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

#include "bondConstraint.hpp"

#include <cmath>
#include <vector>

#include "distanceKernels.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "vector3d.hpp"

using namespace constraints;
using namespace simulationBox;
using namespace linearAlgebra;
using namespace kernel;
using namespace settings;

/**
 * @brief Constructor
 *
 * @param molecule1
 * @param molecule2
 * @param atomIndex1
 * @param atomIndex2
 * @param bondLength
 */
BondConstraint::BondConstraint(
    Molecule    *molecule1,
    Molecule    *molecule2,
    const size_t atomIndex1,
    const size_t atomIndex2,
    const double bondLength
)
    : connectivity::Bond(molecule1, molecule2, atomIndex1, atomIndex2),
      _targetBondLength(bondLength)
{
}

/**
 * @brief calculates the reference bond data of a bond constraint
 *
 * @param simulationBox
 */
void BondConstraint::calculateConstraintBondRef(
    const simulationBox::SimulationBox &simBox
)
{
    simBox.applyPBC(_shakeDistanceRef);

    const auto dxyz = distVec(
        _molecules[0]->getAtomPosition(_atomIndices[0]),
        _molecules[1]->getAtomPosition(_atomIndices[1]),
        simBox
    );

    _shakeDistanceRef = dxyz;
}

/**
 * @brief calculates the distance delta of a bond constraint
 *
 */
double BondConstraint::calculateDistanceDelta(const SimulationBox &simBox) const
{
    const auto pos1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto pos2 = _molecules[1]->getAtomPosition(_atomIndices[1]);

    auto dPosition = pos1 - pos2;
    simBox.applyPBC(dPosition);

    const auto distanceSquared       = normSquared(dPosition);
    const auto targetDistanceSquared = _targetBondLength * _targetBondLength;

    const auto delta = 0.5 * (targetDistanceSquared - distanceSquared);

    return delta;
}

/**
 * @brief applies the shake algorithm to a bond constraint
 *
 * @details if delta is not smaller than tolerance, the shake algorithm is
 * applied
 *
 */
bool BondConstraint::applyShake(
    const SimulationBox &simBox,
    const double         tolerance
)
{
    const auto delta = calculateDistanceDelta(simBox);

    if (std::fabs(delta / (_targetBondLength * _targetBondLength)) > tolerance)
    {
        const auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
        const auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

        const auto sumInvMass              = invMass1 + invMass2;
        const auto shakeDistanceRefSquared = normSquared(_shakeDistanceRef);

        const auto shakeForce = delta / (sumInvMass) / shakeDistanceRefSquared;
        const auto dPosition  = shakeForce * _shakeDistanceRef;

        _molecules[0]->addAtomPosition(_atomIndices[0], +invMass1 * dPosition);
        _molecules[1]->addAtomPosition(_atomIndices[1], -invMass2 * dPosition);

        const auto dVelocity = dPosition / TimingsSettings::getTimeStep();

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
    const auto dVelocity = _molecules[0]->getAtomVelocity(_atomIndices[0]) -
                           _molecules[1]->getAtomVelocity(_atomIndices[1]);

    const auto scalarProduct = dot(dVelocity, _shakeDistanceRef);

    const auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
    const auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

    const auto sumInvMass              = invMass1 + invMass2;
    const auto shakeDistanceRefSquared = normSquared(_shakeDistanceRef);

    const auto delta = -scalarProduct / (sumInvMass) / shakeDistanceRefSquared;

    return delta;
}

/**
 * @brief applies the rattle algorithm to a bond constraint
 *
 * @details if delta is not smaller than tolerance, the rattle algorithm is
 * applied
 *
 */
bool BondConstraint::applyRattle(const double tolerance)
{
    const auto delta = calculateVelocityDelta();

    if (std::fabs(delta) > tolerance)
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

/***************************
 * standard setter methods *
 ***************************/

/**
 * @brief set the shake distance reference
 *
 * @param shakeDistanceRef
 */
void BondConstraint::setShakeDistanceRef(const pq::Vec3D &shakeDistanceRef)
{
    _shakeDistanceRef = shakeDistanceRef;
}

/***************************
 * standard getter methods *
 ***************************/

/**
 * @brief get the target bond length
 *
 * @return target bond length
 */
[[nodiscard]]
double BondConstraint::getTargetBondLength() const
{
    return _targetBondLength;
}

/**
 * @brief get the shake distance reference
 *
 * @return shake distance reference
 */
[[nodiscard]]
pq::Vec3D BondConstraint::getShakeDistanceRef() const
{
    return _shakeDistanceRef;
}