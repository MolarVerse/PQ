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

#include <memory>   // for allocator

#include "constants/conversionFactors.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_
#include "gtest/gtest.h"                     // for Message, TestPartResult
#include "orthorhombicBox.hpp"               // for OrthorhombicBox
#include "vector3d.hpp"                      // for Vec3D

using namespace simulationBox;

TEST(TestOrthoRhombicBox, setBoxDimensions)
{
    auto                       box           = OrthorhombicBox();
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    box.setBoxDimensions(boxDimensions);
    EXPECT_EQ(box.getBoxDimensions(), boxDimensions);
}

TEST(TestOrthoRhombicBox, calculateBoxDimensionsFromDensity)
{
    auto         box       = OrthorhombicBox();
    const double density   = 1.0 / constants::_KG_PER_L_TO_AMU_PER_ANGSTROM3_;
    const double totalMass = 1.0;
    const linearAlgebra::Vec3D boxDimensions = {1.0, 1.0, 1.0};
    EXPECT_EQ(
        box.calculateBoxDimensionsFromDensity(totalMass, density),
        boxDimensions
    );
}

TEST(TestOrthoRhombicBox, calculateVolume)
{
    auto                       box           = OrthorhombicBox();
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    box.setBoxDimensions(boxDimensions);
    EXPECT_EQ(box.calculateVolume(), 6.0);
}

TEST(TestOrthoRhombicBox, applyPeriodicBoundaryConditions)
{
    auto                       box           = OrthorhombicBox();
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    box.setBoxDimensions(boxDimensions);

    linearAlgebra::Vec3D position = {0.5, 1.5, 2.5};
    box.applyPBC(position);

    const linearAlgebra::Vec3D expectedPosition = {0.5, -0.5, -0.5};
    EXPECT_EQ(position, expectedPosition);
}

TEST(TestOrthoRhombicBox, scaleBox)
{
    auto box = OrthorhombicBox();

    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    box.setBoxDimensions(boxDimensions);

    const linearAlgebra::tensor3D scaleFactors =
        diagonalMatrix(linearAlgebra::Vec3D{2.0, 2.0, 2.0});
    box.scaleBox(scaleFactors);

    const linearAlgebra::Vec3D expectedBoxDimensions = {2.0, 4.0, 6.0};
    EXPECT_EQ(box.getBoxDimensions(), expectedBoxDimensions);
}

TEST(TestOrthoRhombicBox, getMinimalBoxDimension)
{
    auto                       box           = OrthorhombicBox();
    const linearAlgebra::Vec3D boxDimensions = {2.0, 1.0, 3.0};
    box.setBoxDimensions(boxDimensions);

    EXPECT_EQ(box.getMinimalBoxDimension(), 1.0);
}

TEST(TestOrthoRhombicBox, calculateShiftVector)
{
    auto box = OrthorhombicBox();

    box.setBoxDimensions({1.0, 1.0, 1.0});
    const linearAlgebra::Vec3D vector{0.2, 1.2, -0.8};

    EXPECT_EQ(box.calculateShiftVector(vector)[0], 0.0);
    EXPECT_EQ(box.calculateShiftVector(vector)[1], 1.0);
    EXPECT_EQ(box.calculateShiftVector(vector)[2], -1.0);
}