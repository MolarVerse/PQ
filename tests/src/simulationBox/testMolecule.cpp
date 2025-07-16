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

#include "testMolecule.hpp"

#include <algorithm>   // for std::ranges::for_each

#include "gtest/gtest.h"         // for Message, TestPartResult
#include "orthorhombicBox.hpp"   // for OrthorhombicBox
#include "staticMatrix.hpp"      // for diagonalMatrix

TEST_F(TestMolecule, calculateCenterOfMass)
{
    const linearAlgebra::Vec3D     boxDimensions = {10.0, 10.0, 10.0};
    const linearAlgebra::Vec3D     centerOfMass  = {1.0 / 3.0, 1.0 / 2.0, 0.0};
    simulationBox::OrthorhombicBox box;
    box.setBoxDimensions(boxDimensions);

    _molecule->calculateCenterOfMass(box);
    EXPECT_EQ(_molecule->getCenterOfMass(), centerOfMass);
}

TEST_F(TestMolecule, scaleAtoms)
{
    // const linearAlgebra::tensor3D scale =
    //     diagonalMatrix(linearAlgebra::Vec3D{1.0, 2.0, 3.0});
    // const linearAlgebra::Vec3D atomPosition1 = _molecule->getAtomPosition(0);
    // const linearAlgebra::Vec3D atomPosition2 = _molecule->getAtomPosition(1);
    // const linearAlgebra::Vec3D atomPosition3 = _molecule->getAtomPosition(2);

    // simulationBox::OrthorhombicBox box;
    // box.setBoxDimensions({10.0, 10.0, 10.0});

    // _molecule->calculateCenterOfMass(box);

    // const auto centerOfMassBeforeScaling = _molecule->getCenterOfMass();
    // const linearAlgebra::Vec3D shift =
    //     centerOfMassBeforeScaling * (diagonal(scale) - 1.0);

    // _molecule->scale(scale, box);

    // EXPECT_EQ(_molecule->getAtomPosition(0), atomPosition1 + shift);
    // EXPECT_EQ(_molecule->getAtomPosition(1), atomPosition2 + shift);
    // EXPECT_EQ(_molecule->getAtomPosition(2), atomPosition3 + shift);

    // Test atoms scaling on the boundaries

    simulationBox::OrthorhombicBox box;
    simulationBox::OrthorhombicBox box_control;
    box.setBoxDimensions({1.0, 2.0, 3.0});
    box_control.setBoxDimensions({1.0, 2.0, 3.0});
    _molecule->setNumberOfAtoms(3);
    _molecule->setAtomPosition(0, linearAlgebra::Vec3D(0.5, 0.0, 0.0));
    _molecule->setAtomPosition(1, linearAlgebra::Vec3D(0.0, 1.5, 0.0));
    _molecule->setAtomPosition(2, linearAlgebra::Vec3D(0.0, 0.0, 2.5));

    EXPECT_EQ(
        _molecule->getAtomPosition(0),
        linearAlgebra::Vec3D(0.5, 0.0, 0.0)
    );
    EXPECT_EQ(
        _molecule->getAtomPosition(1),
        linearAlgebra::Vec3D(0.0, 1.5, 0.0)
    );
    EXPECT_EQ(
        _molecule->getAtomPosition(2),
        linearAlgebra::Vec3D(0.0, 0.0, 2.5)
    );

    const linearAlgebra::tensor3D scale =
        diagonalMatrix(linearAlgebra::Vec3D{-2.0, -2.0, -2.0});

    const linearAlgebra::Vec3D atomPosition_1 = _molecule->getAtomPosition(0);
    const linearAlgebra::Vec3D atomPosition_2 = _molecule->getAtomPosition(1);
    const linearAlgebra::Vec3D atomPosition_3 = _molecule->getAtomPosition(2);

    _molecule->calculateCenterOfMass(box);

    const auto centerOfMassBeforeScaling = _molecule->getCenterOfMass();

    const linearAlgebra::Vec3D shift =
        diagonal(scale) * centerOfMassBeforeScaling - centerOfMassBeforeScaling;

    linearAlgebra::Vec3D pos1 = atomPosition_1 + shift;
    linearAlgebra::Vec3D pos2 = atomPosition_2 + shift;
    linearAlgebra::Vec3D pos3 = atomPosition_3 + shift;

    _molecule->scale(scale, box);

    // scale box
    box_control.scaleBox(scale);

    EXPECT_EQ(_molecule->getAtomPosition(0), pos1);
    EXPECT_EQ(_molecule->getAtomPosition(1), pos2);
    EXPECT_EQ(_molecule->getAtomPosition(2), pos3);
}

TEST_F(TestMolecule, decenterPositions)
{
    simulationBox::OrthorhombicBox box;
    box.setBoxDimensions({1.0, 2.0, 3.0});
    _molecule->setNumberOfAtoms(3);
    _molecule->setAtomPosition(0, linearAlgebra::Vec3D(0.5, 0.0, 0.0));
    _molecule->setAtomPosition(1, linearAlgebra::Vec3D(0.0, 1.5, 0.0));
    _molecule->setAtomPosition(2, linearAlgebra::Vec3D(0.0, 0.0, 2.5));
    const auto centerOfMassBeforeScaling = _molecule->getCenterOfMass();
    const linearAlgebra::tensor3D scale =
        diagonalMatrix(linearAlgebra::Vec3D{-2.0, -2.0, -2.0});
    const linearAlgebra::Vec3D atomPosition_1 = _molecule->getAtomPosition(0);
    const linearAlgebra::Vec3D atomPosition_2 = _molecule->getAtomPosition(1);
    const linearAlgebra::Vec3D atomPosition_3 = _molecule->getAtomPosition(2);

    _molecule->scale(scale, box);
    _molecule->calculateCenterOfMass(box);
    const auto centerOfMassAfterScaling = _molecule->getCenterOfMass();

    _molecule->decenter(box);
    const linearAlgebra::Vec3D shift =
        diagonal(scale) * centerOfMassBeforeScaling - centerOfMassBeforeScaling;

    linearAlgebra::Vec3D pos1 =
        atomPosition_1 + shift + centerOfMassAfterScaling;
    linearAlgebra::Vec3D pos2 =
        atomPosition_2 + shift + centerOfMassAfterScaling;
    linearAlgebra::Vec3D pos3 =
        atomPosition_3 + shift + centerOfMassAfterScaling;

    box.applyPBC(pos1);
    box.applyPBC(pos2);
    box.applyPBC(pos3);

    EXPECT_EQ(_molecule->getAtomPosition(0), pos1);
    EXPECT_EQ(_molecule->getAtomPosition(1), pos2);
    EXPECT_EQ(_molecule->getAtomPosition(2), pos3);
    EXPECT_EQ(_molecule->getCenterOfMass(), centerOfMassAfterScaling);
}

TEST_F(TestMolecule, setAtomForceToZero)
{
    _molecule->setAtomForcesToZero();
    EXPECT_EQ(_molecule->getAtomForce(0), linearAlgebra::Vec3D());
    EXPECT_EQ(_molecule->getAtomForce(1), linearAlgebra::Vec3D());
    EXPECT_EQ(_molecule->getAtomForce(2), linearAlgebra::Vec3D());
}

TEST_F(TestMolecule, getNumberOfAtomTypes)
{
    EXPECT_EQ(_molecule->getNumberOfAtomTypes(), 2);
}