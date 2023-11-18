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

#include "testMolecule.hpp"

#include "orthorhombicBox.hpp"   // for OrthorhombicBox
#include "staticMatrix3x3.hpp"   // for diagonalMatrix

#include "gtest/gtest.h"   // for Message, TestPartResult

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
    const linearAlgebra::tensor3D scale         = diagonalMatrix(linearAlgebra::Vec3D{1.0, 2.0, 3.0});
    const linearAlgebra::Vec3D    atomPosition1 = _molecule->getAtomPosition(0);
    const linearAlgebra::Vec3D    atomPosition2 = _molecule->getAtomPosition(1);
    const linearAlgebra::Vec3D    atomPosition3 = _molecule->getAtomPosition(2);

    simulationBox::OrthorhombicBox box;
    box.setBoxDimensions({10.0, 10.0, 10.0});

    _molecule->calculateCenterOfMass(box);

    const auto                 centerOfMassBeforeScaling = _molecule->getCenterOfMass();
    const linearAlgebra::Vec3D shift                     = centerOfMassBeforeScaling * (diagonal(scale) - 1.0);

    _molecule->scale(scale, box);

    EXPECT_EQ(_molecule->getAtomPosition(0), atomPosition1 + shift);
    EXPECT_EQ(_molecule->getAtomPosition(1), atomPosition2 + shift);
    EXPECT_EQ(_molecule->getAtomPosition(2), atomPosition3 + shift);
}

TEST_F(TestMolecule, setAtomForceToZero)
{
    _molecule->setAtomForcesToZero();
    EXPECT_EQ(_molecule->getAtomForce(0), linearAlgebra::Vec3D());
    EXPECT_EQ(_molecule->getAtomForce(1), linearAlgebra::Vec3D());
    EXPECT_EQ(_molecule->getAtomForce(2), linearAlgebra::Vec3D());
}

TEST_F(TestMolecule, getNumberOfAtomTypes) { EXPECT_EQ(_molecule->getNumberOfAtomTypes(), 2); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
