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

#ifndef _TEST_RING_POLYMER_TRAJECTORY_FILE_OUTPUT_HPP_

#define _TEST_RING_POLYMER_TRAJECTORY_FILE_OUTPUT_HPP_

#include <gtest/gtest.h>   // for Test
#include <stdio.h>         // for remove

#include <algorithm>   // for copy, max
#include <memory>      // for __shared_ptr_access, shared_ptr, make_shared
#include <vector>      // for vector

#include "atom.hpp"                          // for Atom
#include "molecule.hpp"                      // for Molecule
#include "ringPolymerSettings.hpp"           // for RingPolymerSettings
#include "ringPolymerTrajectoryOutput.hpp"   // for RingPolymerTrajectoryOutput
#include "simulationBox.hpp"                 // for SimulationBox
#include "vector3d.hpp"                      // for Vec3D

/**
 * @class TestRingPolymerTrajectoryOutput
 *
 * @brief test suite for ring polymer trajectory output
 *
 */
class TestRingPolymerTrajectoryOutput : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        settings::RingPolymerSettings::setNumberOfBeads(2);

        _trajectoryOutput =
            new output::RingPolymerTrajectoryOutput("default.rpmd.xyz");
        _simulationBox1 = new simulationBox::SimulationBox();
        _simulationBox2 = new simulationBox::SimulationBox();

        _simulationBox1->setBoxDimensions({10.0, 10.0, 10.0});
        _simulationBox2->setBoxDimensions({10.0, 10.0, 10.0});

        auto molecule1_1 = simulationBox::Molecule();
        auto molecule1_2 = simulationBox::Molecule();

        const auto atom1_1 = std::make_shared<simulationBox::Atom>();
        const auto atom2_1 = std::make_shared<simulationBox::Atom>();
        const auto atom1_2 = std::make_shared<simulationBox::Atom>();
        const auto atom2_2 = std::make_shared<simulationBox::Atom>();

        molecule1_1.setNumberOfAtoms(2);
        molecule1_2.setNumberOfAtoms(2);

        atom1_1->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2_1->setPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        atom1_1->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2_1->setForce(linearAlgebra::Vec3D(2.0, 3.0, 4.0));
        atom1_1->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2_1->setVelocity(linearAlgebra::Vec3D(3.0, 4.0, 5.0));
        atom1_1->setName("H");
        atom2_1->setName("O");
        atom1_1->setPartialCharge(1.0);
        atom2_1->setPartialCharge(-1.0);
        molecule1_1.setMoltype(1);
        molecule1_1.addAtom(atom1_1);
        molecule1_1.addAtom(atom2_1);

        atom1_2->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0) + 1.0);
        atom2_2->setPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0) + 1.0);
        atom1_2->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0) + 1.0);
        atom2_2->setForce(linearAlgebra::Vec3D(2.0, 3.0, 4.0) + 1.0);
        atom1_2->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0) + 1.0);
        atom2_2->setVelocity(linearAlgebra::Vec3D(3.0, 4.0, 5.0) + 1.0);
        atom1_2->setName("H");
        atom2_2->setName("O");
        atom1_2->setPartialCharge(1.0 + 1.0);
        atom2_2->setPartialCharge(-1.0 + 1.0);
        molecule1_2.setMoltype(1);
        molecule1_2.addAtom(atom1_2);
        molecule1_2.addAtom(atom2_2);

        auto molecule2_1 = simulationBox::Molecule();
        auto molecule2_2 = simulationBox::Molecule();

        const auto atom3_1 = std::make_shared<simulationBox::Atom>();
        const auto atom3_2 = std::make_shared<simulationBox::Atom>();

        molecule2_1.setNumberOfAtoms(1);
        molecule2_2.setNumberOfAtoms(1);

        atom3_1->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3_1->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3_1->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3_1->setName("Ar");
        atom3_1->setPartialCharge(0.0);
        molecule2_1.setMoltype(2);
        molecule2_1.addAtom(atom3_1);

        atom3_2->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0) + 1.0);
        atom3_2->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0) + 1.0);
        atom3_2->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0) + 1.0);
        atom3_2->setName("Ar");
        atom3_2->setPartialCharge(0.0 + 1.0);
        molecule2_2.setMoltype(2);
        molecule2_2.addAtom(atom3_2);

        _simulationBox1->addMolecule(molecule1_1);
        _simulationBox1->addMolecule(molecule2_1);
        _simulationBox1->addAtom(atom1_1);
        _simulationBox1->addAtom(atom2_1);
        _simulationBox1->addAtom(atom3_1);

        _simulationBox2->addMolecule(molecule1_2);
        _simulationBox2->addMolecule(molecule2_2);
        _simulationBox2->addAtom(atom1_2);
        _simulationBox2->addAtom(atom2_2);
        _simulationBox2->addAtom(atom3_2);

        _simulationBox1->setNumberOfAtoms(3);
        _simulationBox2->setNumberOfAtoms(3);

        _beads.push_back(*_simulationBox1);
        _beads.push_back(*_simulationBox2);
    }

    void TearDown() override
    {
        delete _trajectoryOutput;
        delete _simulationBox1;
        delete _simulationBox2;
        ::remove("default.rpmd.xyz");
    }

    output::RingPolymerTrajectoryOutput      *_trajectoryOutput;
    simulationBox::SimulationBox             *_simulationBox1;
    simulationBox::SimulationBox             *_simulationBox2;
    std::vector<simulationBox::SimulationBox> _beads;
};

#endif   // _TEST_RING_POLYMER_TRAJECTORY_FILE_OUTPUT_HPP_