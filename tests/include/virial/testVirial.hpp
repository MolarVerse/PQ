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

#ifndef _TEST_VIRIAL_HPP_

#define _TEST_VIRIAL_HPP_

#include <gtest/gtest.h>

#include <memory>

#include "atom.hpp"
#include "enums/jobtype.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "virial.hpp"

class TestVirial : public ::testing::Test
{
   protected:
    std::unique_ptr<molsys::SimulationBox> _simBox;
    JobType                                _jobType;

    void SetUp() override
    {
        _jobType = settings::GeneralSettings::getJobtype();

        _simBox = std::make_unique<molsys::SimulationBox>();
        settings::GeneralSettings::setVirialType(VirialType::ATOMIC);

        auto molecule1 = molsys::Molecule();

        const auto atom1 = std::make_shared<molsys::Atom>();
        const auto atom2 = std::make_shared<molsys::Atom>();

        atom1->setPosition(linalg::Vec3D(1.0, 1.0, 1.0));
        atom2->setPosition(linalg::Vec3D(1.0, 2.0, 3.0));
        atom1->setForce(linalg::Vec3D(1.0, 1.0, 1.0));
        atom2->setForce(linalg::Vec3D(1.0, 2.0, 3.0));
        atom1->setShiftForce(linalg::Vec3D(1.0, 1.0, 1.0));
        atom2->setShiftForce(linalg::Vec3D(1.0, 2.0, 3.0));
        molecule1.setCenterOfMass(linalg::Vec3D(1.0, 1.0, 1.0));
        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        auto molecule2 = molsys::Molecule();

        auto atom3 = std::make_shared<molsys::Atom>();

        atom3->setPosition(linalg::Vec3D(1.0, 1.0, 1.0));
        atom3->setForce(linalg::Vec3D(1.0, 1.0, 1.0));
        atom3->setShiftForce(linalg::Vec3D(1.0, 1.0, 1.0));
        molecule2.setCenterOfMass(linalg::Vec3D(0.0, 0.0, 0.0));
        molecule2.addAtom(atom3);

        _simBox->addMolecule(molecule1);
        _simBox->addMolecule(molecule2);

        _simBox->addAtom(atom1);
        _simBox->addAtom(atom2);
        _simBox->addAtom(atom3);

        _simBox->setBoxDimensions(linalg::Vec3D(10.0, 10.0, 10.0));
    }

    void TearDown() override
    {
        settings::GeneralSettings::setJobtype(_jobType);
    }
};

#endif
