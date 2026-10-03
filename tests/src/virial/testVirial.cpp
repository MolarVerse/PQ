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

#include <gtest/gtest.h>

#include "virial.hpp"

TEST_F(TestVirial, calculateVirial)
{
    const auto &molecule0 = _simBox->getMolecule(0);
    const auto &molecule1 = _simBox->getMolecule(1);

    const auto force_mol1_atom1 = molecule0.getAtomForce(AtomIndex{0});
    const auto force_mol1_atom2 = molecule0.getAtomForce(AtomIndex{1});
    const auto force_mol2_atom1 = molecule1.getAtomForce(AtomIndex{0});

    const auto position_mol1_atom1 = molecule0.getAtomPosition(AtomIndex{0});
    const auto position_mol1_atom2 = molecule0.getAtomPosition(AtomIndex{1});
    const auto position_mol2_atom1 = molecule1.getAtomPosition(AtomIndex{0});

    const auto shiftForce_mol1_atom1 = molecule0.getAtomShiftForce(0);
    const auto shiftForce_mol1_atom2 = molecule0.getAtomShiftForce(1);
    const auto shiftForce_mol2_atom1 = molecule1.getAtomShiftForce(0);

    const auto virial = force_mol1_atom1 * position_mol1_atom1 +
                        force_mol1_atom2 * position_mol1_atom2 +
                        force_mol2_atom1 * position_mol2_atom1 +
                        shiftForce_mol1_atom1 + shiftForce_mol1_atom2 +
                        shiftForce_mol2_atom1;

    const auto virialCalc = virial::calculateVirial(*_simBox);
    EXPECT_EQ(diagonal(virialCalc), virial);
    EXPECT_EQ(_simBox->getMolecule(0).getAtomShiftForce(0), linalg::Vec3D{0});
    EXPECT_EQ(_simBox->getMolecule(0).getAtomShiftForce(1), linalg::Vec3D{0});
    EXPECT_EQ(_simBox->getMolecule(1).getAtomShiftForce(0), linalg::Vec3D{0});
}

TEST_F(TestVirial, atomicVirialHasNoIntramolecularCorrection)
{
    EXPECT_EQ(
        virial::intraMolecularVirialCorrection(*_simBox),
        linalg::tensor3D{0.0}
    );
}

TEST_F(TestVirial, intramolecularCorrection)
{
    const auto &molecule0 = _simBox->getMolecule(0);
    const auto &molecule1 = _simBox->getMolecule(1);

    const auto force_mol1_atom1 = molecule0.getAtomForce(AtomIndex{0});
    const auto force_mol1_atom2 = molecule0.getAtomForce(AtomIndex{1});
    const auto force_mol2_atom1 = molecule1.getAtomForce(AtomIndex{0});

    const auto position_mol1_atom1 = molecule0.getAtomPosition(AtomIndex{0});
    const auto position_mol1_atom2 = molecule0.getAtomPosition(AtomIndex{1});
    const auto position_mol2_atom1 = molecule1.getAtomPosition(AtomIndex{0});

    const auto shiftForce_mol1_atom1 = molecule0.getAtomShiftForce(0);
    const auto shiftForce_mol1_atom2 = molecule0.getAtomShiftForce(1);
    const auto shiftForce_mol2_atom1 = molecule1.getAtomShiftForce(0);

    auto virial = force_mol1_atom1 * position_mol1_atom1 +
                  force_mol1_atom2 * position_mol1_atom2 +
                  force_mol2_atom1 * position_mol2_atom1 +
                  shiftForce_mol1_atom1 + shiftForce_mol1_atom2 +
                  shiftForce_mol2_atom1;

    const auto virialCalc = virial::calculateVirial(*_simBox);

    EXPECT_EQ(diagonal(virialCalc), virial);
}

TEST_F(TestVirial, calculateQMVirialWithNoQMAtomsIsZero)
{
    // an MM-only jobtype makes every atom non-QM regardless of its
    // isActive() state
    settings::Settings::setJobtype(JobType::MM_MD);

    EXPECT_EQ(virial::calculateQMVirial(*_simBox), linalg::tensor3D{0.0});
}

TEST_F(TestVirial, calculateQMVirialSumsOnlyQMAtomContributions)
{
    settings::Settings::setJobtype(JobType::QM_MD);

    const auto &molecule0 = _simBox->getMolecule(0);
    const auto &molecule1 = _simBox->getMolecule(1);

    const auto force_mol1_atom1 = molecule0.getAtomForce(AtomIndex{0});
    const auto force_mol1_atom2 = molecule0.getAtomForce(AtomIndex{1});
    const auto force_mol2_atom1 = molecule1.getAtomForce(AtomIndex{0});

    const auto position_mol1_atom1 = molecule0.getAtomPosition(AtomIndex{0});
    const auto position_mol1_atom2 = molecule0.getAtomPosition(AtomIndex{1});
    const auto position_mol2_atom1 = molecule1.getAtomPosition(AtomIndex{0});

    // a QM-only jobtype makes every atom a QM atom, so the expected virial
    // is the same tensor-product sum as the atomic virial, but WITHOUT any
    // shift-force contribution (calculateQMVirial does not add shift
    // forces)
    const auto virial = force_mol1_atom1 * position_mol1_atom1 +
                        force_mol1_atom2 * position_mol1_atom2 +
                        force_mol2_atom1 * position_mol2_atom1;

    EXPECT_EQ(diagonal(virial::calculateQMVirial(*_simBox)), virial);
}

TEST_F(TestVirial, calculateMolecularVirial)
{
    settings::Settings::setVirialType(VirialType::MOLECULAR);

    const auto &molecule0 = _simBox->getMolecule(0);
    const auto &molecule1 = _simBox->getMolecule(1);

    const auto force_mol1_atom1 = molecule0.getAtomForce(AtomIndex{0});
    const auto force_mol1_atom2 = molecule0.getAtomForce(AtomIndex{1});
    const auto force_mol2_atom1 = molecule1.getAtomForce(AtomIndex{0});

    const auto position_mol1_atom1 = molecule0.getAtomPosition(AtomIndex{0});
    const auto position_mol1_atom2 = molecule0.getAtomPosition(AtomIndex{1});
    const auto position_mol2_atom1 = molecule1.getAtomPosition(AtomIndex{0});

    const auto centerOfMass_mol1 = molecule0.getCenterOfMass();
    const auto centerOfMass_mol2 = molecule1.getCenterOfMass();

    const auto virial =
        -force_mol1_atom1 * (position_mol1_atom1 - centerOfMass_mol1) -
        force_mol1_atom2 * (position_mol1_atom2 - centerOfMass_mol1) -
        force_mol2_atom1 * (position_mol2_atom1 - centerOfMass_mol2);

    const auto virialCalculated =
        virial::intraMolecularVirialCorrection(*_simBox);

    EXPECT_EQ(diagonal(virialCalculated), virial);
}
