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

#include "bondForceField.hpp"

#include <vector>   // for vector

#include "coulombPotential.hpp"   // for CoulombPotential
#include "forceField.hpp"         // for correctLinker
#include "molecule.hpp"           // for Molecule
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "vector3d.hpp"           // for Vector3D, norm, operator*, Vec3D

using namespace forceField;
using namespace simulationBox;
using namespace connectivity;
using namespace linearAlgebra;
using namespace physicalData;
using namespace potential;

using enum HybridZone;

/**
 * @brief constructor
 *
 * @param molecule1
 * @param molecule2
 * @param atomIndex1
 * @param atomIndex2
 * @param type
 */
BondForceField::BondForceField(
    Molecule    *molecule1,
    Molecule    *molecule2,
    const size_t atomIndex1,
    const size_t atomIndex2,
    const size_t type
)
    : Bond(molecule1, molecule2, atomIndex1, atomIndex2), _type(type)
{
}

/**
 * @brief calculate energy and forces for a single bond
 *
 * @details if bond is a linker bond, correct coulomb and non-coulomb energy and
 * forces
 *
 * @param box
 * @param physicalData
 */
void BondForceField::calculateEnergyAndForces(
    const SimulationBox    &box,
    PhysicalData           &physicalData,
    const CoulombPotential &coulombPotential,
    NonCoulombPotential    &nonCoulombPotential
)
{
    const bool bothInactive =
        !_molecules[0]->isActive() && !_molecules[1]->isActive();

    if (bothInactive)
        return;

    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    auto       dPosition = position1 - position2;

    box.applyPBC(dPosition);

    const auto distance      = norm(dPosition);
    const auto deltaDistance = distance - _equilBondLength;

    auto forceMagnitude = -_forceConstant * deltaDistance;

    physicalData.addBondEnergy(-forceMagnitude * deltaDistance / 2.0);

    if (_isLinker && distance < CoulombPotential::getCoulombRadiusCutOff())
    {
        forceMagnitude += correctLinker<BondForceField>(
            coulombPotential,
            nonCoulombPotential,
            physicalData,
            _molecules[0],
            _molecules[1],
            _atomIndices[0],
            _atomIndices[1],
            distance
        );
    }

    forceMagnitude /= distance;

    const auto force = forceMagnitude * dPosition;

    _molecules[0]->addAtomForce(_atomIndices[0], force);
    _molecules[1]->addAtomForce(_atomIndices[1], -force);

    auto smF = 0.0;
    if (_molecules[0]->getHybridZone() == SMOOTHING)
        smF = _molecules[0]->getSmoothingFactor();

    physicalData.addVirial(tensorProduct(dPosition, force) * (1 - smF));
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set if bond is a linker bond
 *
 * @param isLinker
 */
void BondForceField::setIsLinker(const bool isLinker) { _isLinker = isLinker; }

/**
 * @brief set equilibrium bond length
 *
 * @param equilibriumBondLength
 */
void BondForceField::setEquilibriumBondLength(
    const double equilibriumBondLength
)
{
    _equilBondLength = equilibriumBondLength;
}

/**
 * @brief set force constant
 *
 * @param forceConstant
 */
void BondForceField::setForceConstant(const double forceConstant)
{
    _forceConstant = forceConstant;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get if bond is a linker bond
 *
 * @return true
 * @return false
 */
bool BondForceField::isLinker() const { return _isLinker; }

/**
 * @brief get the type of the bond
 *
 * @return size_t
 */
size_t BondForceField::getType() const { return _type; }

/**
 * @brief get the equilibrium bond length
 *
 * @return double
 */
double BondForceField::getEquilibriumBondLength() const
{
    return _equilBondLength;
}

/**
 * @brief get the force constant
 *
 * @return double
 */
double BondForceField::getForceConstant() const { return _forceConstant; }