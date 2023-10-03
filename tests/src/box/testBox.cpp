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

#include "testBox.hpp"

#include "constants/conversionFactors.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_
#include "orthorhombicBox.hpp"               // for OrthorhombicBox
#include "vector3d.hpp"                      // for Vec3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <memory>          // for allocator

using namespace simulationBox;

TEST_F(TestBox, setBoxDimensions)
{
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->getBoxDimensions(), boxDimensions);
}

TEST_F(TestBox, calculateBoxDimensionsFromDensity)
{
    const double               density       = 1.0 / constants::_KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_;
    const double               totalMass     = 1.0;
    const linearAlgebra::Vec3D boxDimensions = {1.0, 1.0, 1.0};
    EXPECT_EQ(_box->calculateBoxDimensionsFromDensity(totalMass, density), boxDimensions);
}

TEST_F(TestBox, calculateVolume)
{
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box                                     = new simulationBox::OrthorhombicBox();
    _box->setBoxDimensions(boxDimensions);
    EXPECT_EQ(_box->calculateVolume(), 6.0);
}

TEST_F(TestBox, applyPeriodicBoundaryConditions)
{
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    linearAlgebra::Vec3D position = {0.5, 1.5, 2.5};
    _box->applyPBC(position);

    const linearAlgebra::Vec3D expectedPosition = {0.5, -0.5, -0.5};
    EXPECT_EQ(position, expectedPosition);
}

TEST_F(TestBox, scaleBox)
{
    _box = new simulationBox::OrthorhombicBox();

    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    const linearAlgebra::Vec3D scaleFactors = {2.0, 2.0, 2.0};
    _box->scaleBox(scaleFactors);

    const linearAlgebra::Vec3D expectedBoxDimensions = {2.0, 4.0, 6.0};
    EXPECT_EQ(_box->getBoxDimensions(), expectedBoxDimensions);
}

TEST_F(TestBox, getMinimalBoxDimension)
{
    const linearAlgebra::Vec3D boxDimensions = {2.0, 1.0, 3.0};
    _box->setBoxDimensions(boxDimensions);

    EXPECT_EQ(_box->getMinimalBoxDimension(), 1.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}