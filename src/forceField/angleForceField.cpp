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

#include "angleForceField.hpp"

#include <cmath>   // for sqrt, sin

#include "coulombPotential.hpp"   // for CoulombPotential
#include "forceField.hpp"         // for correctLinker
#include "molecule.hpp"           // for Molecule
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "vector3d.hpp"           // for Vector3D, cross, operator*, normSquared

using namespace forceField;
using namespace simulationBox;
using namespace connectivity;
using namespace linearAlgebra;
using namespace physicalData;
using namespace potential;

/**
 * @brief constructor
 *
 * @param molecules
 * @param atomIndices
 * @param type
 */
AngleForceField::AngleForceField(
    const std::vector<Molecule *> &molecules,
    const std::vector<size_t>     &atomIndices,
    const size_t                   type
)
    : Angle(molecules, atomIndices), _type(type){};

/**
 * @brief calculate energy and forces for a single alpha
 *
 * @details if angle is a linker angle, correct coulomb and non-coulomb energy
 * and forces
 *
 * @param box
 * @param physicalData
 */
void AngleForceField::calculateEnergyAndForces(
    const SimulationBox    &box,
    PhysicalData           &physicalData,
    const CoulombPotential &coulombPotential,
    NonCoulombPotential    &nonCoulombPotential
)
{
    // central position of alpha
    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    const auto position3 = _molecules[2]->getAtomPosition(_atomIndices[2]);

    auto dPosition12 = position1 - position2;
    auto dPosition13 = position1 - position3;

    box.applyPBC(dPosition12);
    box.applyPBC(dPosition13);

    const auto distance12Squared = normSquared(dPosition12);
    const auto distance13Squared = normSquared(dPosition13);

    const auto distance12 = ::sqrt(distance12Squared);
    const auto distance13 = ::sqrt(distance13Squared);

    const auto alpha      = angle(dPosition12, dPosition13);
    const auto deltaAngle = alpha - _equilibriumAngle;

    auto forceMagnitude = -_forceConstant * deltaAngle;

    physicalData.addAngleEnergy(-forceMagnitude * deltaAngle / 2.0);

    const auto normalDistance = distance12 * distance13 * ::sin(alpha);

    auto normalPosition  = cross(dPosition13, dPosition12);
    normalPosition      /= normalDistance;

    auto force    = forceMagnitude / distance12Squared;
    auto forcexyz = force * cross(dPosition12, normalPosition);

    _molecules[0]->addAtomForce(_atomIndices[0], -forcexyz);
    _molecules[1]->addAtomForce(_atomIndices[1], forcexyz);

    force    = forceMagnitude / distance13Squared;
    forcexyz = force * cross(normalPosition, dPosition13);

    _molecules[0]->addAtomForce(_atomIndices[0], -forcexyz);
    _molecules[2]->addAtomForce(_atomIndices[2], forcexyz);

    if (_isLinker)
    {
        auto dPosition23 = position2 - position3;
        box.applyPBC(dPosition23);

        const auto distance23 = norm(dPosition23);

        if (distance23 < CoulombPotential::getCoulombRadiusCutOff())
        {
            forceMagnitude = correctLinker<AngleForceField>(
                coulombPotential,
                nonCoulombPotential,
                physicalData,
                _molecules[1],
                _molecules[2],
                _atomIndices[1],
                _atomIndices[2],
                distance23
            );

            forceMagnitude /= distance23;

            forcexyz = forceMagnitude * dPosition23;

            physicalData.addVirial(tensorProduct(dPosition23, forcexyz));

            _molecules[1]->addAtomForce(_atomIndices[1], forcexyz);
            _molecules[2]->addAtomForce(_atomIndices[2], -forcexyz);
        }
    }
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set if angle is a linker angle
 *
 * @param isLinker
 */
void AngleForceField::setIsLinker(const bool isLinker) { _isLinker = isLinker; }

/**
 * @brief set equilibrium angle
 *
 * @param equilibriumAngle
 */
void AngleForceField::setEquilibriumAngle(const double equilibriumAngle)
{
    _equilibriumAngle = equilibriumAngle;
}

/**
 * @brief set force constant
 *
 * @param forceConstant
 */
void AngleForceField::setForceConstant(const double forceConstant)
{
    _forceConstant = forceConstant;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get if angle is a linker angle
 *
 * @return true
 * @return false
 */
bool AngleForceField::isLinker() const { return _isLinker; }

/**
 * @brief get type of angle
 *
 * @return size_t
 */
size_t AngleForceField::getType() const { return _type; }

/**
 * @brief get equilibrium angle
 *
 * @return double
 */
double AngleForceField::getEquilibriumAngle() const
{
    return _equilibriumAngle;
}

/**
 * @brief get force constant
 *
 * @return double
 */
double AngleForceField::getForceConstant() const { return _forceConstant; }