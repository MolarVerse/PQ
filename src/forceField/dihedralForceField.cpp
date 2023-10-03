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

#include "dihedralForceField.hpp"

#include "coulombPotential.hpp"   // for CoulombPotential
#include "forceField.hpp"         // for correctLinker
#include "molecule.hpp"           // for Molecule
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "vector3d.hpp"           // for operator*, Vector3D, dot, cross, norm

#include <cmath>   // for cos, sin

using namespace forceField;

/**
 * @brief calculate energy and forces for a single dihedral
 *
 * @details if dihedral is a linker dihedral, correct coulomb and non-coulomb energy and forces (only for non improper dihedrals)
 *
 * @param box
 * @param physicalData
 */
void DihedralForceField::calculateEnergyAndForces(const simulationBox::SimulationBox &box,
                                                  physicalData::PhysicalData         &physicalData,
                                                  const bool                          isImproperDihedral,
                                                  const potential::CoulombPotential  &coulombPotential,
                                                  potential::NonCoulombPotential     &nonCoulombPotential)
{
    const auto position2 = _molecules[1]->getAtomPosition(_atomIndices[1]);   // central position of dihedral
    const auto position3 = _molecules[2]->getAtomPosition(_atomIndices[2]);   // central position of dihedral

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

    if (isImproperDihedral)
        physicalData.addImproperEnergy(_forceConstant * (1.0 + ::cos(_periodicity * phi + _phaseShift)));
    else
        physicalData.addDihedralEnergy(_forceConstant * (1.0 + ::cos(_periodicity * phi + _phaseShift)));

    auto       forceMagnitude = distance23 / distance123Squared;
    const auto forceVector12  = forceMagnitude * crossPosition123;

    forceMagnitude           = distance23 / distance432Squared;
    const auto forceVector43 = forceMagnitude * crossPosition432;

    forceMagnitude            = dot(dPosition12, dPosition23) / (distance123Squared * distance23);
    const auto forceVector123 = forceMagnitude * crossPosition123;

    forceMagnitude            = dot(dPosition43, dPosition23) / (distance432Squared * distance23);
    const auto forceVector432 = forceMagnitude * crossPosition432;

    forceMagnitude = _forceConstant * _periodicity * ::sin(_periodicity * phi + _phaseShift);

    _molecules[0]->addAtomForce(_atomIndices[0], -forceMagnitude * forceVector12);
    _molecules[1]->addAtomForce(_atomIndices[1], +forceMagnitude * (forceVector12 + forceVector123 - forceVector432));
    _molecules[2]->addAtomForce(_atomIndices[2], +forceMagnitude * (-forceVector43 - forceVector123 + forceVector432));
    _molecules[3]->addAtomForce(_atomIndices[3], +forceMagnitude * forceVector43);

    if (_isLinker)
    {
        auto dPosition14 = position1 - position4;
        box.applyPBC(dPosition14);

        const auto distance14 = norm(dPosition14);

        if (distance14 < potential::CoulombPotential::getCoulombRadiusCutOff())
        {
            forceMagnitude = correctLinker<DihedralForceField>(coulombPotential,
                                                               nonCoulombPotential,
                                                               physicalData,
                                                               _molecules[0],
                                                               _molecules[3],
                                                               _atomIndices[0],
                                                               _atomIndices[3],
                                                               distance14);

            forceMagnitude /= distance14;

            const auto forcexyz = forceMagnitude * dPosition14;

            if (!isImproperDihedral)
                physicalData.addVirial(forcexyz * dPosition14);

            _molecules[0]->addAtomForce(_atomIndices[0], forcexyz);
            _molecules[3]->addAtomForce(_atomIndices[3], -forcexyz);
        }
    }
}
