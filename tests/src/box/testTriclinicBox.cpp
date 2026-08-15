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

#include <array>

#include "constants/conversionFactors.hpp"   // for _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_
#include "defaults.hpp"                      // for VACUUM_BOX_DIMENSION
#include "gtest/gtest.h"                     // for Message, TestPartResult
#include "manostatSettings.hpp"   // for ManostatSettings
#include "matrixNear.hpp"         // for EXPECT_MATRIX_NEAR
#include "triclinicBox.hpp"       // for TriclinicBox
#include "vectorNear.hpp"         // for EXPECT_VECTOR_NEAR

using namespace molsys;

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

    const auto alpha = 30.0 * constants::DEG_TO_RAD;
    const auto beta  = 60.0 * constants::DEG_TO_RAD;
    const auto gamma = 45.0 * constants::DEG_TO_RAD;

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

    const auto alpha = 30.0 * constants::DEG_TO_RAD;
    const auto beta  = 60.0 * constants::DEG_TO_RAD;
    const auto gamma = 45.0 * constants::DEG_TO_RAD;

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

TEST(TestTriclinicBox, transformsRoundTrip)
{
    TriclinicBox box;
    box.setBoxDimensions({4.0, 5.0, 6.0});
    box.setBoxAngles({80.0, 75.0, 70.0});

    const linearAlgebra::Vec3D vector{1.0, 2.0, 3.0};
    const auto                 tensor = linearAlgebra::tensor3D(
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    );

    EXPECT_VECTOR_NEAR(
        box.toSimSpace(box.toOrthoSpace(vector)),
        vector,
        1.0e-12
    );
    EXPECT_MATRIX_NEAR(
        box.toSimSpace(box.toOrthoSpace(tensor)),
        tensor,
        1.0e-12
    );
    EXPECT_NEAR(box.cosAlpha(), std::cos(80.0 * constants::DEG_TO_RAD), 1e-15);
    EXPECT_NEAR(box.cosBeta(), std::cos(75.0 * constants::DEG_TO_RAD), 1e-15);
    EXPECT_NEAR(box.cosGamma(), std::cos(70.0 * constants::DEG_TO_RAD), 1e-15);
    EXPECT_NEAR(box.sinAlpha(), std::sin(80.0 * constants::DEG_TO_RAD), 1e-15);
    EXPECT_NEAR(box.sinBeta(), std::sin(75.0 * constants::DEG_TO_RAD), 1e-15);
    EXPECT_NEAR(box.sinGamma(), std::sin(70.0 * constants::DEG_TO_RAD), 1e-15);
    EXPECT_GT(box.getMinimalBoxDimension(), 0.0);
}

TEST(TestTriclinicBox, periodicityMasksBoxMatrix)
{
    TriclinicBox box;
    box.setBoxDimensions({4.0, 5.0, 6.0});
    box.setBoxAngles({80.0, 75.0, 70.0});

    using enum Periodicity;
    constexpr std::array periodicities{NON_PERIODIC, X, Y, Z, XY, XZ, YZ, XYZ};
    for (const auto periodicity : periodicities)
        EXPECT_TRUE(std::isfinite(box.getBoxMatrix(periodicity)[0][0]));

    const auto nonPeriodic = box.getBoxMatrix(NON_PERIODIC);
    EXPECT_DOUBLE_EQ(nonPeriodic[0][0], defaults::VACUUM_BOX_DIMENSION);
    EXPECT_DOUBLE_EQ(nonPeriodic[1][1], defaults::VACUUM_BOX_DIMENSION);
    EXPECT_DOUBLE_EQ(nonPeriodic[2][2], defaults::VACUUM_BOX_DIMENSION);
    EXPECT_DOUBLE_EQ(nonPeriodic[0][1], 0.0);
    EXPECT_DOUBLE_EQ(nonPeriodic[0][2], 0.0);
    EXPECT_EQ(box.getBoxMatrix(XYZ), box.getBoxMatrix());
}

TEST(TestTriclinicBox, scalingPreservesConsistentDimensionsAndAngles)
{
    TriclinicBox box;
    box.setBoxDimensions({4.0, 5.0, 6.0});
    box.setBoxAngles({80.0, 75.0, 70.0});

    settings::ManostatSettings::setIsotropy(settings::Isotropy::ISOTROPIC);
    box.scaleBox(diagonalMatrix(linearAlgebra::Vec3D{2.0, 2.0, 2.0}));
    EXPECT_VECTOR_NEAR(
        box.getBoxDimensions(),
        linearAlgebra::Vec3D(8.0, 10.0, 12.0),
        1.0e-12
    );
    EXPECT_NEAR(box.getVolume(), box.calculateVolume(), 1.0e-12);

    settings::ManostatSettings::setIsotropy(
        settings::Isotropy::FULL_ANISOTROPIC
    );
    const auto originalAngles = box.getBoxAngles();
    box.scaleBox(diagonalMatrix(linearAlgebra::Vec3D{0.5, 0.5, 0.5}));
    EXPECT_VECTOR_NEAR(
        box.getBoxDimensions(),
        linearAlgebra::Vec3D(4.0, 5.0, 6.0),
        1.0e-12
    );
    EXPECT_VECTOR_NEAR(box.getBoxAngles(), originalAngles, 1.0e-12);

    const auto [dimensions, angles] =
        calcBoxDimAndAnglesFromBoxMatrix(box.getBoxMatrix());
    EXPECT_VECTOR_NEAR(dimensions, box.getBoxDimensions(), 1.0e-12);
    EXPECT_VECTOR_NEAR(angles, box.getBoxAngles(), 1.0e-12);
}
