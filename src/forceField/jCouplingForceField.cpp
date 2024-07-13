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

#include "jCouplingForceField.hpp"

#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimBox

using namespace forceField;
using namespace connectivity;
using namespace physicalData;
using namespace simulationBox;
using namespace potential;

/**
 * @brief Construct a new JCouplingForceField::JCouplingForceField object
 *
 * @param molecules
 * @param atomIndices
 * @param type
 */
JCouplingForceField::JCouplingForceField(
    const std::vector<pq::Molecule *> &molecules,
    const std::vector<size_t>         &atomIndices,
    const size_t                       type
)
    : Dihedral(molecules, atomIndices), _type(type)
{
}

/**
 * @brief calculate energy and forces for a single dihedral
 *
 * @param simBox
 * @param data
 */
void JCouplingForceField::calculateEnergyAndForces(
    const SimulationBox &simBox,
    PhysicalData        &data
)
{
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    const auto position3 = _molecules[2]->getAtomPosition(_atomIndices[2]);

    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position4 = _molecules[3]->getAtomPosition(_atomIndices[3]);

    auto dPosition12 = position1 - position2;
    auto dPosition23 = position2 - position3;
    auto dPosition43 = position4 - position3;

    simBox.applyPBC(dPosition12);
    simBox.applyPBC(dPosition23);
    simBox.applyPBC(dPosition43);

    const auto crossPosition123 = cross(dPosition12, dPosition23);
    const auto crossPosition432 = cross(dPosition43, dPosition23);

    const auto distance123Squared = normSquared(crossPosition123);
    const auto distance432Squared = normSquared(crossPosition432);

    const auto distance23 = norm(dPosition23);

    auto phi = angle(crossPosition123, crossPosition432);
    phi      = dot(dPosition12, crossPosition432) > 0.0 ? -phi : phi;

    const auto cosine = ::cos(phi);
    const auto J_phi  = _a * cosine * cosine + _b * cosine + _c;
    const auto deltaJ = J_phi - _J0;

    if (deltaJ < 0.0 && !_lowerSymmetry)
        return;
    else if (deltaJ > 0.0 && !_upperSymmetry)
        return;

    const auto energy = 0.5 * _forceConstant * deltaJ * deltaJ;

    data.addJCouplingEnergy(energy);

    auto       forceMagnitude = distance23 / distance123Squared;
    const auto forceVector12  = forceMagnitude * crossPosition123;

    forceMagnitude           = distance23 / distance432Squared;
    const auto forceVector43 = forceMagnitude * crossPosition432;

    forceMagnitude             = dot(dPosition12, dPosition23);
    forceMagnitude            /= (distance123Squared * distance23);
    const auto forceVector123  = forceMagnitude * crossPosition123;

    forceMagnitude             = dot(dPosition43, dPosition23);
    forceMagnitude            /= (distance432Squared * distance23);
    const auto forceVector432  = forceMagnitude * crossPosition432;

    const auto sine         = ::sin(phi);
    const auto energyFactor = _forceConstant * deltaJ;
    const auto J_derivative = 2 * _a * cosine + _b;
    forceMagnitude          = sine * energyFactor * J_derivative;

    const auto diffForce123_432 = forceVector123 - forceVector432;

    const auto force_0 = -forceMagnitude * forceVector12;
    const auto force_1 = forceMagnitude * (forceVector12 + diffForce123_432);
    const auto force_2 = +forceMagnitude * (-forceVector43 - diffForce123_432);
    const auto force_3 = forceMagnitude * forceVector43;

    _molecules[0]->addAtomForce(_atomIndices[0], force_0);
    _molecules[1]->addAtomForce(_atomIndices[1], force_1);
    _molecules[2]->addAtomForce(_atomIndices[2], force_2);
    _molecules[3]->addAtomForce(_atomIndices[3], force_3);
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief Set the upper symmetry
 *
 * @param boolean
 */
void JCouplingForceField::setUpperSymmetry(const bool boolean)
{
    _upperSymmetry = boolean;
}

/**
 * @brief Set the lower symmetry
 *
 * @param boolean
 */
void JCouplingForceField::setLowerSymmetry(const bool boolean)
{
    _lowerSymmetry = boolean;
}

/**
 * @brief Set the J0
 *
 * @param J0
 */
void JCouplingForceField::setJ0(const double J0) { _J0 = J0; }

/**
 * @brief Set the force constant
 *
 * @param k
 */
void JCouplingForceField::setForceConstant(const double k)
{
    _forceConstant = k;
}

/**
 * @brief Set the a
 *
 * @param a
 */
void JCouplingForceField::setA(const double a) { _a = a; }

/**
 * @brief Set the b
 *
 * @param b
 */
void JCouplingForceField::setB(const double b) { _b = b; }

/**
 * @brief Set the c
 *
 * @param c
 */
void JCouplingForceField::setC(const double c) { _c = c; }

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the type
 *
 * @return size_t
 */
size_t JCouplingForceField::getType() const { return _type; }

/**
 * @brief get if the upper symmetry is set
 *
 * @return bool
 */
bool JCouplingForceField::getUpperSymmetry() const { return _upperSymmetry; }

/**
 * @brief get if the lower symmetry is set
 *
 * @return bool
 */
bool JCouplingForceField::getLowerSymmetry() const { return _lowerSymmetry; }

/**
 * @brief get the J0
 *
 * @return double
 */
double JCouplingForceField::getJ0() const { return _J0; }

/**
 * @brief get the force constant
 *
 * @return double
 */
double JCouplingForceField::getForceConstant() const { return _forceConstant; }

/**
 * @brief get the a
 *
 * @return double
 */
double JCouplingForceField::getA() const { return _a; }

/**
 * @brief get the b
 *
 * @return double
 */
double JCouplingForceField::getB() const { return _b; }

/**
 * @brief get the c
 *
 * @return double
 */
double JCouplingForceField::getC() const { return _c; }