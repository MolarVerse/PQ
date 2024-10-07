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

#include "testVirial.hpp"

#include <memory>   // for allocator

#include "gtest/gtest.h"         // for Message, TestPartResult
#include "molecularVirial.hpp"   // for MolecularVirial

using namespace linearAlgebra;
using namespace physicalData;
using namespace virial;

TEST_F(TestVirial, calculateVirial)
{
    const auto &molecule0 = _simBox->getMolecule(0);
    const auto &molecule1 = _simBox->getMolecule(1);

    const auto force_mol1_atom1 = molecule0.getAtomForce(0);
    const auto force_mol1_atom2 = molecule0.getAtomForce(1);
    const auto force_mol2_atom1 = molecule1.getAtomForce(0);

    const auto position_mol1_atom1 = molecule0.getAtomPosition(0);
    const auto position_mol1_atom2 = molecule0.getAtomPosition(1);
    const auto position_mol2_atom1 = molecule1.getAtomPosition(0);

    const auto shiftForce_mol1_atom1 = molecule0.getAtomShiftForce(0);
    const auto shiftForce_mol1_atom2 = molecule0.getAtomShiftForce(1);
    const auto shiftForce_mol2_atom1 = molecule1.getAtomShiftForce(0);

    const auto virial = force_mol1_atom1 * position_mol1_atom1 +
                        force_mol1_atom2 * position_mol1_atom2 +
                        force_mol2_atom1 * position_mol2_atom1 +
                        shiftForce_mol1_atom1 + shiftForce_mol1_atom2 +
                        shiftForce_mol2_atom1;

    _virial->calculateVirial(*_simBox, *_data);

    EXPECT_EQ(diagonal(_data->getVirial()), virial);
    EXPECT_EQ(_simBox->getMolecule(0).getAtomShiftForce(0), Vec3D{0});
    EXPECT_EQ(_simBox->getMolecule(0).getAtomShiftForce(1), Vec3D{0});
    EXPECT_EQ(_simBox->getMolecule(1).getAtomShiftForce(0), Vec3D{0});
}

TEST_F(TestVirial, intramolecularCorrection)
{
    auto *virialClass = new MolecularVirial();
    virialClass->setVirial({0.0});

    const auto &molecule0 = _simBox->getMolecule(0);
    const auto &molecule1 = _simBox->getMolecule(1);

    const auto force_mol1_atom1 = molecule0.getAtomForce(0);
    const auto force_mol1_atom2 = molecule0.getAtomForce(1);
    const auto force_mol2_atom1 = molecule1.getAtomForce(0);

    const auto position_mol1_atom1 = molecule0.getAtomPosition(0);
    const auto position_mol1_atom2 = molecule0.getAtomPosition(1);
    const auto position_mol2_atom1 = molecule1.getAtomPosition(0);

    const auto shiftForce_mol1_atom1 = molecule0.getAtomShiftForce(0);
    const auto shiftForce_mol1_atom2 = molecule0.getAtomShiftForce(1);
    const auto shiftForce_mol2_atom1 = molecule1.getAtomShiftForce(0);

    auto virial = force_mol1_atom1 * position_mol1_atom1 +
                  force_mol1_atom2 * position_mol1_atom2 +
                  force_mol2_atom1 * position_mol2_atom1 +
                  shiftForce_mol1_atom1 + shiftForce_mol1_atom2 +
                  shiftForce_mol2_atom1;

    virialClass->calculateVirial(*_simBox, *_data);

    EXPECT_EQ(diagonal(_data->getVirial()), virial);
}

TEST_F(TestVirial, calculateMolecularVirial)
{
    auto *virialClass = new MolecularVirial();
    virialClass->setVirial(tensor3D(0.0));

    const auto &molecule0 = _simBox->getMolecule(0);
    const auto &molecule1 = _simBox->getMolecule(1);

    const auto force_mol1_atom1 = molecule0.getAtomForce(0);
    const auto force_mol1_atom2 = molecule0.getAtomForce(1);
    const auto force_mol2_atom1 = molecule1.getAtomForce(0);

    const auto position_mol1_atom1 = molecule0.getAtomPosition(0);
    const auto position_mol1_atom2 = molecule0.getAtomPosition(1);
    const auto position_mol2_atom1 = molecule1.getAtomPosition(0);

    const auto centerOfMass_mol1 = molecule0.getCenterOfMass();
    const auto centerOfMass_mol2 = molecule1.getCenterOfMass();

    const auto virial =
        -force_mol1_atom1 * (position_mol1_atom1 - centerOfMass_mol1) -
        force_mol1_atom2 * (position_mol1_atom2 - centerOfMass_mol1) -
        force_mol2_atom1 * (position_mol2_atom1 - centerOfMass_mol2);

    PhysicalData physicalData;

    virialClass->intraMolecularVirialCorrection(*_simBox, physicalData);

    EXPECT_EQ(diagonal(virialClass->getVirial()), virial);
}