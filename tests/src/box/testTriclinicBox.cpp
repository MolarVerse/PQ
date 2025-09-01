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
#include "matrixNear.hpp"                    // for EXPECT_MATRIX_NEAR
#include "staticMatrix.hpp"                  // for StaticMatrix3x3
#include "triclinicBox.hpp"                  // for TriclinicBox
#include "vector3d.hpp"                      // for Vec3D
#include "vectorNear.hpp"                    // for EXPECT_VECTOR_NEAR

using namespace simulationBox;

TEST(TestTriclinicBox, setBoxDimensions)
{
    auto                       box           = TriclinicBox();
    const linearAlgebra::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    box.setBoxDimensions(boxDimensions);
    EXPECT_EQ(box.getBoxDimensions(), boxDimensions);

    // only first entry should be set to boxDimensions[0] because angles are not
    // set yet
    EXPECT_EQ(
        box.getBoxMatrix(),
        linearAlgebra::StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0}
        )
    );
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
        box.getBoxMatrix(),
        linearAlgebra::StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 2.0, 0.0},
            {0.0, 0.0, 3.0}
        ),
        1e-15
    );
    EXPECT_MATRIX_NEAR(
        box.getTransformationMatrix(),
        linearAlgebra::StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        ),
        1e-15
    );

    box.setBoxAngles({30.0, 60.0, 45.0});

    const auto alpha = 30.0 * constants::_DEG_TO_RAD_;
    const auto beta  = 60.0 * constants::_DEG_TO_RAD_;
    const auto gamma = 45.0 * constants::_DEG_TO_RAD_;

    EXPECT_MATRIX_NEAR(
        box.getTransformationMatrix(),
        linearAlgebra::StaticMatrix3x3<double>(
            {1.0, sqrt(0.5), ::cos(beta)},
            {0.0,
             sqrt(0.5),
             (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)},
            {0.0,
             0.0,
             ::sqrt(
                 1 - cos(alpha) * cos(alpha) - cos(beta) * cos(beta) -
                 cos(gamma) * cos(gamma) +
                 2 * cos(alpha) * cos(beta) * cos(gamma)
             ) / sin(gamma)}
        ),
        1e-15
    );

    auto boxMatrix = linearAlgebra::StaticMatrix3x3<double>();
    boxMatrix[0] = {box.getTransformationMatrix()[0] * box.getBoxDimensions()};
    boxMatrix[1] = {box.getTransformationMatrix()[1] * box.getBoxDimensions()};
    boxMatrix[2] = {box.getTransformationMatrix()[2] * box.getBoxDimensions()};
    EXPECT_MATRIX_NEAR(box.getBoxMatrix(), boxMatrix, 1.0e-15);
}

TEST(TestTriclinicBox, calculateVolume)
{
    auto box = TriclinicBox();
    box.setBoxDimensions({1.0, 2.0, 3.0});
    box.setBoxAngles({30.0, 60.0, 45.0});

    const auto alpha = 30.0 * constants::_DEG_TO_RAD_;
    const auto beta  = 60.0 * constants::_DEG_TO_RAD_;
    const auto gamma = 45.0 * constants::_DEG_TO_RAD_;

    const auto volume =
        1.0 * 2.0 * 3.0 *
        ::sqrt(
            1 - ::cos(alpha) * ::cos(alpha) - ::cos(beta) * ::cos(beta) -
            ::cos(gamma) * ::cos(gamma) +
            2 * ::cos(alpha) * ::cos(beta) * ::cos(gamma)
        );

    EXPECT_DOUBLE_EQ(box.calculateVolume(), volume);
}

TEST(TestTriclinicBox, applyPBC)
{
    auto box = TriclinicBox();
    box.setBoxDimensions({1.0, 2.0, 3.0});
    box.setBoxAngles({30.0, 60.0, 45.0});

    auto position = linearAlgebra::Vec3D({1.3, 2.3, 3.3});

    box.applyPBC(position);

    EXPECT_VECTOR_NEAR(
        position,
        linearAlgebra::Vec3D(
            {0.12842712474619078, 0.77995789639665647, 0.45556413851582972}
        ),
        1e-8
    );
}

TEST(TestTriclinicBox, calculateShiftVectors)
{
    auto box = TriclinicBox();
    box.setBoxDimensions({1.0, 2.0, 3.0});
    box.setBoxAngles({30.0, 60.0, 45.0});

    const auto position    = linearAlgebra::Vec3D({1.3, 2.3, 3.3});
    const auto newPosition = linearAlgebra::Vec3D(
        {0.12842712474619078, 0.77995789639665647, 0.45556413851582972}
    );

    const auto shiftVector = box.calcShiftVector(position);

    EXPECT_VECTOR_NEAR(shiftVector, (position - newPosition), 1e-8);
}

TEST(TestTriclinicBox, wrapPositionIntoBox)
{
    auto box = TriclinicBox();
    box.setBoxDimensions({60.0, 60.0, 4.542});
    box.setBoxAngles({90.0, 90.0, 120.0});

    auto outsidePos = linearAlgebra::Vec3D({5.0, -30.0, -0.1});

    box.applyPBC(outsidePos);
    EXPECT_VECTOR_NEAR(
        outsidePos,
        linearAlgebra::Vec3D(5.0, -30.0, -0.1),
        1e-10
    );

    const auto wrappedPos = box.wrapPositionIntoBox(outsidePos);
    EXPECT_VECTOR_NEAR(
        wrappedPos,
        linearAlgebra::Vec3D(-25.0, 21.96152422706632, -0.1),
        1e-10
    );
}