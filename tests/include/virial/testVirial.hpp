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

#ifndef TESTS__INCLUDE__VIRIAL__TEST_VIRIAL_HPP
#define TESTS__INCLUDE__VIRIAL__TEST_VIRIAL_HPP

#include <gtest/gtest.h>

#include <memory>

#include "atom.hpp"
#include "enums/jobtype.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "units.hpp"
#include "virial.hpp"

class TestVirial : public ::testing::Test
{
   protected:
    std::unique_ptr<molsys::SimulationBox> _simBox;
    JobType                                _jobType;

    void SetUp() override;

    void TearDown() override;
};

inline void TestVirial::SetUp()
{
    _simBox = std::make_unique<molsys::SimulationBox>();
    GeneralSettings::setVirialType(VirialType::ATOMIC);

    auto molecule1 = molsys::Molecule();

    const auto atom1 = std::make_shared<molsys::Atom>();
    const auto atom2 = std::make_shared<molsys::Atom>();

    atom1->setPosition(linalg::Vec3D(1.0, 1.0, 1.0) * angstrom);
    atom2->setPosition(linalg::Vec3D(1.0, 2.0, 3.0) * angstrom);
    atom1->setForce(linalg::Vec3D(1.0, 1.0, 1.0) * kcal_per_mol / angstrom);
    atom2->setForce(linalg::Vec3D(1.0, 2.0, 3.0) * kcal_per_mol / angstrom);
    atom1->setShiftForce(linalg::Vec3D(1.0, 1.0, 1.0) * kcal_per_mol);
    atom2->setShiftForce(linalg::Vec3D(1.0, 2.0, 3.0) * kcal_per_mol);
    molecule1.setCenterOfMass(
        Pos3D{1.0 * angstrom, 1.0 * angstrom, 1.0 * angstrom}
    );
    molecule1.addAtom(atom1);
    molecule1.addAtom(atom2);

    auto molecule2 = molsys::Molecule();

    auto atom3 = std::make_shared<molsys::Atom>();

    atom3->setPosition(linalg::Vec3D(1.0, 1.0, 1.0) * angstrom);
    atom3->setForce(linalg::Vec3D(1.0, 1.0, 1.0) * kcal_per_mol / angstrom);
    atom3->setShiftForce(linalg::Vec3D(1.0, 1.0, 1.0) * kcal_per_mol);
    molecule2.setCenterOfMass(Pos3D{Pos::zero()});
    molecule2.addAtom(atom3);

    _simBox->addMolecule(molecule1);
    _simBox->addMolecule(molecule2);

    _simBox->addAtom(atom1);
    _simBox->addAtom(atom2);
    _simBox->addAtom(atom3);

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    _simBox->setBox(orthoBox, false);
}

inline void TestVirial::TearDown() {}

#endif
