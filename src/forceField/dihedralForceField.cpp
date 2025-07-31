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

#include "dihedralForceField.hpp"

#include <cmath>   // for cos, sin

#include "coulombPotential.hpp"   // for CoulombPotential
#include "forceField.hpp"         // for correctLinker
#include "molecule.hpp"           // for Molecule
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "vector3d.hpp"           // for operator*, Vector3D, dot, cross, norm

using namespace forceField;
using namespace connectivity;
using namespace linearAlgebra;
using namespace physicalData;
using namespace potential;
using namespace simulationBox;

/**
 * @brief Construct a new Dihedral Force Field:: Dihedral Force Field object
 *
 * @param molecules
 * @param atomIndices
 * @param type
 */
DihedralForceField::DihedralForceField(
    const std::vector<Molecule *> &molecules,
    const std::vector<size_t>     &atomIndices,
    const size_t                   type
)
    : Dihedral(molecules, atomIndices), _type(type)
{
}

/**
 * @brief calculate energy and forces for a single dihedral
 *
 * @details if dihedral is a linker dihedral, correct coulomb and non-coulomb
 * energy and forces (only for non improper dihedrals)
 *
 * @param box
 * @param physicalData
 */
void DihedralForceField::calculateEnergyAndForces(
    const SimulationBox    &box,
    PhysicalData           &physicalData,
    const bool              isImproperDihedral,
    const CoulombPotential &coulombPotential,
    NonCoulombPotential    &nonCoulombPotential
)
{
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);
    const auto position3 = _molecules[2]->getAtomPosition(_atomIndices[2]);

    const auto position1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto position4 = _molecules[3]->getAtomPosition(_atomIndices[3]);

    auto dPosition12 = position1 - position2;
    auto dPosition23 = position2 - position3;
    auto dPosition43 = position4 - position3;

    box.applyPBC(dPosition12);
    box.applyPBC(dPosition23);
    box.applyPBC(dPosition43);

    const auto crossPosition123 = cross(dPosition12, dPosition23);
    const auto crossPosition432 = cross(dPosition43, dPosition23);

    const auto distance123Squared = normSquared(crossPosition123);
    const auto distance432Squared = normSquared(crossPosition432);

    const auto distance23 = norm(dPosition23);

    auto phi = angle(crossPosition123, crossPosition432);
    phi      = dot(dPosition12, crossPosition432) > 0.0 ? -phi : phi;

    const auto cosine = ::cos(_periodicity * phi + _phaseShift);
    const auto energy = _forceConstant * (1.0 + cosine);

    if (isImproperDihedral)
        physicalData.addImproperEnergy(energy);
    else
        physicalData.addDihedralEnergy(energy);

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

    const auto sine = ::sin(_periodicity * phi + _phaseShift);
    forceMagnitude  = _forceConstant * _periodicity * sine;

    const auto diffForce123_432 = forceVector123 - forceVector432;

    const auto force_0 = -forceMagnitude * forceVector12;
    const auto force_1 = forceMagnitude * (forceVector12 + diffForce123_432);
    const auto force_2 = +forceMagnitude * (-forceVector43 - diffForce123_432);
    const auto force_3 = forceMagnitude * forceVector43;

    _molecules[0]->addAtomForce(_atomIndices[0], force_0);
    _molecules[1]->addAtomForce(_atomIndices[1], force_1);
    _molecules[2]->addAtomForce(_atomIndices[2], force_2);
    _molecules[3]->addAtomForce(_atomIndices[3], force_3);

    if (_isLinker)
    {
        auto dPosition14 = position1 - position4;
        box.applyPBC(dPosition14);

        const auto distance14 = norm(dPosition14);

        if (distance14 < CoulombPotential::getCoulombRadiusCutOff())
        {
            forceMagnitude = correctLinker<DihedralForceField>(
                coulombPotential,
                nonCoulombPotential,
                physicalData,
                _molecules[0],
                _molecules[3],
                _atomIndices[0],
                _atomIndices[3],
                distance14
            );

            forceMagnitude /= distance14;

            const auto forcexyz = forceMagnitude * dPosition14;

            if (!isImproperDihedral)
                physicalData.addVirial(tensorProduct(dPosition14, forcexyz));

            _molecules[0]->addAtomForce(_atomIndices[0], forcexyz);
            _molecules[3]->addAtomForce(_atomIndices[3], -forcexyz);
        }
    }
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set if dihedral is a linker dihedral
 *
 * @param isLinker
 */
void DihedralForceField::setIsLinker(const bool isLinker)
{
    _isLinker = isLinker;
}

/**
 * @brief set force constant
 *
 * @param forceConstant
 */
void DihedralForceField::setForceConstant(const double forceConstant)
{
    _forceConstant = forceConstant;
}

/**
 * @brief set periodicity
 *
 * @param periodicity
 */
void DihedralForceField::setPeriodicity(const double periodicity)
{
    _periodicity = periodicity;
}

/**
 * @brief set phase shift
 *
 * @param phaseShift
 */
void DihedralForceField::setPhaseShift(const double phaseShift)
{
    _phaseShift = phaseShift;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get if dihedral is a linker dihedral
 *
 * @return true
 * @return false
 */
bool DihedralForceField::isLinker() const { return _isLinker; }

/**
 * @brief get type of dihedral
 *
 * @return size_t
 */
size_t DihedralForceField::getType() const { return _type; }

/**
 * @brief get force constant
 *
 * @return double
 */
double DihedralForceField::getForceConstant() const { return _forceConstant; }

/**
 * @brief get periodicity
 *
 * @return double
 */
double DihedralForceField::getPeriodicity() const { return _periodicity; }

/**
 * @brief get phase shift
 *
 * @return double
 */
double DihedralForceField::getPhaseShift() const { return _phaseShift; }