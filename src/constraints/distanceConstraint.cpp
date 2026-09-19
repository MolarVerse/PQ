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

#include "distanceConstraint.hpp"

#include <cmath>
#include <vector>

#include "simulationBox.hpp"

using namespace constraints;
using namespace molsys;
using namespace connectivity;

/**
 * @brief Construct a new Distance Constraint:: Distance Constraint object
 *
 * @param molecule1
 * @param molecule2
 * @param atomIndex1
 * @param atomIndex2
 * @param lowerDistance
 * @param upperDistance
 * @param springConstant
 * @param dSpringConstantDt
 */
DistanceConstraint::DistanceConstraint(
    Molecule *molecule1,
    Molecule *molecule2,
    AtomIndex atomIndex1,
    AtomIndex atomIndex2,
    double    lowerDistance,
    double    upperDistance,
    double    springConstant,
    double    dSpringConstantDt
)
    : Bond(molecule1, molecule2, atomIndex1, atomIndex2),
      _lowerDistance(lowerDistance),
      _upperDistance(upperDistance),
      _springConstant(springConstant),
      _dSpringConstantDt(dSpringConstantDt)
{
}

/**
 * @brief calculates the reference distance of all distance constraints
 *
 * @param simulationBox
 * @param timeInterval the time interval over which the constraint is applied
 *
 */
void DistanceConstraint::applyDistanceConstraint(
    const molsys::SimulationBox &simulationBox,
    double                       timeInterval
)
{
    _lowerEnergy = 0.0;
    _upperEnergy = 0.0;
    linearAlgebra::Vec3D force;

    if (timeInterval < 0.0)
        return;

    const auto pos1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto pos2 = _molecules[1]->getAtomPosition(_atomIndices[1]);

    auto dPos = pos2 - pos1;
    simulationBox.applyPBC(dPos);

    const auto distance = norm(dPos);
    const auto force_constant =
        _springConstant + (_dSpringConstantDt * timeInterval);

    if (distance < _lowerDistance)
    {
        const auto delta = _lowerDistance - distance;
        _lowerEnergy     = 0.5 * force_constant * delta * delta;
        force            = -force_constant * delta * dPos / distance;
    }
    else if (distance > _upperDistance)
    {
        const auto delta = distance - _upperDistance;
        _upperEnergy     = 0.5 * force_constant * delta * delta;
        force            = +force_constant * delta * dPos / distance;
    }
    else
    {
        return;
    }

    _molecules[0]->addAtomForce(_atomIndices[0], force);
    _molecules[1]->addAtomForce(_atomIndices[1], -force);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get lower distance
 *
 * @return double
 */
double DistanceConstraint::getLowerDistance() const { return _lowerDistance; }

/**
 * @brief get upper distance
 *
 * @return double
 */
double DistanceConstraint::getUpperDistance() const { return _upperDistance; }

/**
 * @brief get spring constant
 *
 * @return double
 */
double DistanceConstraint::getSpringConstant() const { return _springConstant; }

/**
 * @brief get dSpringConstantDt
 *
 * @return double
 */
double DistanceConstraint::getDSpringConstantDt() const
{
    return _dSpringConstantDt;
}

/**
 * @brief get lower energy
 *
 * @return double
 */
double DistanceConstraint::getLowerEnergy() const { return _lowerEnergy; }

/**
 * @brief get upper energy
 *
 * @return double
 */
double DistanceConstraint::getUpperEnergy() const { return _upperEnergy; }
