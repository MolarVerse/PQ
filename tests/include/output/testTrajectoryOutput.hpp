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

#ifndef _TEST_TRAJECTORY_FILE_OUTPUT_HPP_

#define _TEST_TRAJECTORY_FILE_OUTPUT_HPP_

#include <gtest/gtest.h>   // for Test

#include <memory>   // for __shared_ptr_access, shared_ptr, make_shared

#include "atom.hpp"               // for Atom
#include "molecule.hpp"           // for Molecule
#include "simulationBox.hpp"      // for SimulationBox
#include "trajectoryOutput.hpp"   // for TrajectoryOutput

/**
 * @class TestTrajectoryOutput
 *
 * @brief test suite for trajectory output
 *
 */
class TestTrajectoryOutput : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        _trajectoryOutput = new out::TrajectoryOutput("default.xyz");
        _simulationBox    = new molsys::SimulationBox();

        _simulationBox->setBoxDimensions({10.0, 10.0, 10.0});

        auto molecule1 = molsys::Molecule();

        const auto atom1 = std::make_shared<molsys::Atom>();
        const auto atom2 = std::make_shared<molsys::Atom>();

        molecule1.setNumberOfAtoms(2);

        atom1->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        atom1->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setForce(linearAlgebra::Vec3D(2.0, 3.0, 4.0));
        atom1->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setVelocity(linearAlgebra::Vec3D(3.0, 4.0, 5.0));
        atom1->setName("H");
        atom2->setName("O");
        atom1->setPartialCharge(1.0);
        atom2->setPartialCharge(-1.0);
        molecule1.setMoltype(1);
        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        auto molecule2 = molsys::Molecule();

        auto atom3 = std::make_shared<molsys::Atom>();

        molecule2.setNumberOfAtoms(1);

        atom3->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3->setName("Ar");
        atom3->setPartialCharge(0.0);
        molecule2.setMoltype(2);
        molecule2.addAtom(atom3);

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);
        _simulationBox->addAtom(atom1);
        _simulationBox->addAtom(atom2);
        _simulationBox->addAtom(atom3);
    }

    void TearDown() override
    {
        delete _trajectoryOutput;
        delete _simulationBox;
        const auto errorCode = std::remove("default.xyz");
        EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.xyz";
    }

    out::TrajectoryOutput *_trajectoryOutput;
    molsys::SimulationBox *_simulationBox;
};

#endif   // _TEST_TRAJECTORY_FILE_OUTPUT_HPP_
