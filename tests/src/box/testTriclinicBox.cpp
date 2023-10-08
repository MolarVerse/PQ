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

#include "constants/conversionFactors.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_
#include "matrixNear.hpp"                    // for EXPECT_MATRIX_NEAR
#include "staticMatrix3x3.hpp"               // for StaticMatrix3x3
#include "triclinicBox.hpp"                  // for TriclinicBox
#include "vector3d.hpp"                      // for Vec3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <memory>          // for allocator

using namespace simulationBox;

TEST(TestTriclinicBox, setBoxDimensions)
{
    auto                       box           = TriclinicBox();
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    box.setBoxDimensions(boxDimensions);
    EXPECT_EQ(box.getBoxDimensions(), boxDimensions);

    // only first entry should be set to boxDimensions[0] because angles are not set yet
    EXPECT_EQ(box.getBoxMatrix(), linearAlgebra::StaticMatrix3x3<double>({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}));
}

TEST(TestTriclinicBox, setBoxAngles)
{
    auto                       box           = TriclinicBox();
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    const linearAlgebra::Vec3D boxAngles     = {90.0, 90.0, 90.0};
    box.setBoxDimensions(boxDimensions);
    box.setBoxAngles(boxAngles);
    EXPECT_EQ(box.getBoxDimensions(), boxDimensions);
    EXPECT_EQ(box.getBoxAngles(), boxAngles);
    EXPECT_MATRIX_NEAR(
        box.getBoxMatrix(), linearAlgebra::StaticMatrix3x3<double>({1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}), 1e-15);
    EXPECT_MATRIX_NEAR(box.getTransformationMatrix(),
                       linearAlgebra::StaticMatrix3x3<double>({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}),
                       1e-15);

    box.setBoxAngles({30.0, 60.0, 45.0});

    EXPECT_MATRIX_NEAR(box.getTransformationMatrix(),
                       linearAlgebra::StaticMatrix3x3<double>(
                           {1.0, sqrt(0.5), sqrt(3) / 2.0}, {0.0, sqrt(0.5), sqrt(3.0) * sqrt(0.5) / 2.0}, {0.0, 0.0, 1.0}),
                       1e-15);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}