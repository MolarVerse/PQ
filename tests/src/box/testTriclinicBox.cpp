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

#include <gtest/gtest.h>

#include <array>

#include "constants/conversionFactors.hpp"
#include "defaults.hpp"
#include "manostatSettings.hpp"
#include "matrixNear.hpp"
#include "staticMatrix/staticMatrix3x3Class.hpp"
#include "triclinicBox.hpp"
#include "vectorNear.hpp"

struct TestTriclinicBox
{
    [[nodiscard]]
    static double cosAlpha(const linalg::Vec3D &boxAngles)
    {
        return molsys::TriclinicBox::_cosAlpha(
            (boxAngles * mstd::units::deg).in(mstd::units::rad)
        );
    }
    [[nodiscard]]
    static double cosBeta(const linalg::Vec3D &boxAngles)
    {
        return molsys::TriclinicBox::_cosBeta(
            (boxAngles * mstd::units::deg).in(mstd::units::rad)
        );
    }
    [[nodiscard]]
    static double cosGamma(const linalg::Vec3D &boxAngles)
    {
        return molsys::TriclinicBox::_cosGamma(
            (boxAngles * mstd::units::deg).in(mstd::units::rad)
        );
    }
    [[nodiscard]]
    static double sinAlpha(const linalg::Vec3D &boxAngles)
    {
        return molsys::TriclinicBox::_sinAlpha(
            (boxAngles * mstd::units::deg).in(mstd::units::rad)
        );
    }
    [[nodiscard]]
    static double sinBeta(const linalg::Vec3D &boxAngles)
    {
        return molsys::TriclinicBox::_sinBeta(
            (boxAngles * mstd::units::deg).in(mstd::units::rad)
        );
    }
    [[nodiscard]]
    static double sinGamma(const linalg::Vec3D &boxAngles)
    {
        return molsys::TriclinicBox::_sinGamma(
            (boxAngles * mstd::units::deg).in(mstd::units::rad)
        );
    }

    [[nodiscard]]
    static linalg::tensor3D getBoxMatrix(molsys::TriclinicBox &box)
    {
        return box._boxMatrix.get(box);
    }

    static void setBoxAngles(
        molsys::TriclinicBox &box,
        const linalg::Vec3D  &boxAngles
    )
    {
        box._set_angles((boxAngles * mstd::units::deg).in(mstd::units::rad));
    }

    static linalg::tensor3D getTransformationMatrix(molsys::TriclinicBox &box)
    {
        return box._trafoMatrix.get(box);
    }
};

TEST(TestTriclinicBox, setBoxDimensions)
{
    const linalg::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    auto box = molsys::TriclinicBox(boxDimensions, {90.0, 90.0, 90.0});

    EXPECT_MATRIX_NEAR(
        TestTriclinicBox::getBoxMatrix(box),
        linalg::StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 2.0, 0.0},
            {0.0, 0.0, 3.0}
        ),
        1e-15
    );
}

TEST(TestTriclinicBox, setBoxAngles)
{
    const linalg::Vec3D boxDimensions = {1.0, 2.0, 3.0};
    const linalg::Vec3D boxAngles     = {90.0, 90.0, 90.0};
    auto                box = molsys::TriclinicBox(boxDimensions, boxAngles);
    EXPECT_EQ(box.get_boxDimensions(), boxDimensions);
    EXPECT_EQ(box.getBoxAngles(), boxAngles);
    EXPECT_MATRIX_NEAR(
        TestTriclinicBox::getBoxMatrix(box),
        linalg::StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 2.0, 0.0},
            {0.0, 0.0, 3.0}
        ),
        1e-15
    );
    EXPECT_MATRIX_NEAR(
        TestTriclinicBox::getTransformationMatrix(box),
        linalg::StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        ),
        1e-15
    );

    TestTriclinicBox::setBoxAngles(box, {30.0, 60.0, 45.0});

    const auto alpha = 30.0 * DEG_TO_RAD;
    const auto beta  = 60.0 * DEG_TO_RAD;
    const auto gamma = 45.0 * DEG_TO_RAD;

    EXPECT_MATRIX_NEAR(
        TestTriclinicBox::getTransformationMatrix(box),
        linalg::StaticMatrix3x3<double>(
            {1.0, sqrt(0.5), ::cos(beta)},
            {0.0,
             sqrt(0.5),
             (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)},
            {0.0,
             0.0,
             ::sqrt(
                 1 - (cos(alpha) * cos(alpha)) - (cos(beta) * cos(beta)) -
                 (cos(gamma) * cos(gamma)) +
                 (2 * cos(alpha) * cos(beta) * cos(gamma))
             ) / sin(gamma)}
        ),
        1e-15
    );

    auto        boxMatrix   = linalg::StaticMatrix3x3<double>();
    const auto  trafoMatrix = TestTriclinicBox::getTransformationMatrix(box);
    const auto &boxDims     = box.get_boxDimensions();
    boxMatrix[0]            = {trafoMatrix[0] * boxDims};
    boxMatrix[1]            = {trafoMatrix[1] * boxDims};
    boxMatrix[2]            = {trafoMatrix[2] * boxDims};
    EXPECT_MATRIX_NEAR(TestTriclinicBox::getBoxMatrix(box), boxMatrix, 1.0e-15);
}

TEST(TestTriclinicBox, calculateVolume)
{
    auto box = molsys::TriclinicBox({1.0, 2.0, 3.0}, {30.0, 60.0, 45.0});

    const auto alpha = 30.0 * DEG_TO_RAD;
    const auto beta  = 60.0 * DEG_TO_RAD;
    const auto gamma = 45.0 * DEG_TO_RAD;

    const auto volume =
        1.0 * 2.0 * 3.0 *
        ::sqrt(
            1 - (::cos(alpha) * ::cos(alpha)) - (::cos(beta) * ::cos(beta)) -
            (::cos(gamma) * ::cos(gamma)) +
            (2 * ::cos(alpha) * ::cos(beta) * ::cos(gamma))
        );

    EXPECT_DOUBLE_EQ(box.getVolume(), volume);
}

TEST(TestTriclinicBox, applyPBC)
{
    auto box = molsys::TriclinicBox({1.0, 2.0, 3.0}, {30.0, 60.0, 45.0});

    auto position = linalg::Vec3D({1.3, 2.3, 3.3});

    box.applyPBC(position);

    EXPECT_VECTOR_NEAR(
        position,
        linalg::Vec3D(
            {0.12842712474619078, 0.77995789639665647, 0.45556413851582972}
        ),
        1e-8
    );
}

TEST(TestTriclinicBox, calculateShiftVectors)
{
    auto box = molsys::TriclinicBox({1.0, 2.0, 3.0}, {30.0, 60.0, 45.0});

    const auto position    = linalg::Vec3D({1.3, 2.3, 3.3});
    const auto newPosition = linalg::Vec3D(
        {0.12842712474619078, 0.77995789639665647, 0.45556413851582972}
    );

    const auto shiftVector = box.calcShiftVector(position);

    EXPECT_VECTOR_NEAR(shiftVector, (position - newPosition), 1e-8);
}

TEST(TestTriclinicBox, wrapPositionIntoBox)
{
    auto box = molsys::TriclinicBox({60.0, 60.0, 4.542}, {90.0, 90.0, 120.0});

    auto outsidePos = linalg::Vec3D({5.0, -30.0, -0.1});

    box.applyPBC(outsidePos);
    EXPECT_VECTOR_NEAR(outsidePos, linalg::Vec3D(5.0, -30.0, -0.1), 1e-10);

    const auto wrappedPos = box.wrapPositionIntoBox(outsidePos);
    EXPECT_VECTOR_NEAR(
        wrappedPos,
        linalg::Vec3D(-25.0, 21.96152422706632, -0.1),
        1e-10
    );
}

TEST(TestTriclinicBox, transformsRoundTrip)
{
    molsys::TriclinicBox box({4.0, 5.0, 6.0}, {80.0, 75.0, 70.0});

    const linalg::Vec3D vector{1.0, 2.0, 3.0};
    const auto          tensor =
        linalg::tensor3D({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0});

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
    EXPECT_NEAR(
        TestTriclinicBox::cosAlpha(box.getBoxAngles()),
        std::cos(80.0 * DEG_TO_RAD),
        1e-15
    );
    EXPECT_NEAR(
        TestTriclinicBox::cosBeta(box.getBoxAngles()),
        std::cos(75.0 * DEG_TO_RAD),
        1e-15
    );
    EXPECT_NEAR(
        TestTriclinicBox::cosGamma(box.getBoxAngles()),
        std::cos(70.0 * DEG_TO_RAD),
        1e-15
    );
    EXPECT_NEAR(
        TestTriclinicBox::sinAlpha(box.getBoxAngles()),
        std::sin(80.0 * DEG_TO_RAD),
        1e-15
    );
    EXPECT_NEAR(
        TestTriclinicBox::sinBeta(box.getBoxAngles()),
        std::sin(75.0 * DEG_TO_RAD),
        1e-15
    );
    EXPECT_NEAR(
        TestTriclinicBox::sinGamma(box.getBoxAngles()),
        std::sin(70.0 * DEG_TO_RAD),
        1e-15
    );
    EXPECT_GT(box.getMinimalBoxDimension(), 0.0);
}

TEST(TestTriclinicBox, periodicityMasksBoxMatrix)
{
    molsys::TriclinicBox box({4.0, 5.0, 6.0}, {80.0, 75.0, 70.0});

    using enum molsys::Periodicity;
    constexpr std::array periodicities{NON_PERIODIC, X, Y, Z, XY, XZ, YZ, XYZ};
    for (const auto periodicity : periodicities)
        EXPECT_TRUE(std::isfinite(box.getBoxMatrix(periodicity)[0][0]));

    const auto nonPeriodic = box.getBoxMatrix(NON_PERIODIC);
    EXPECT_DOUBLE_EQ(nonPeriodic[0][0], defaults::VACUUM_BOX_DIMENSION);
    EXPECT_DOUBLE_EQ(nonPeriodic[1][1], defaults::VACUUM_BOX_DIMENSION);
    EXPECT_DOUBLE_EQ(nonPeriodic[2][2], defaults::VACUUM_BOX_DIMENSION);
    EXPECT_DOUBLE_EQ(nonPeriodic[0][1], 0.0);
    EXPECT_DOUBLE_EQ(nonPeriodic[0][2], 0.0);
    EXPECT_EQ(box.getBoxMatrix(XYZ), TestTriclinicBox::getBoxMatrix(box));
}

TEST(TestTriclinicBox, scalingPreservesConsistentDimensionsAndAngles)
{
    molsys::TriclinicBox box({4.0, 5.0, 6.0}, {80.0, 75.0, 70.0});

    settings::ManostatSettings::setIsotropy(Isotropy::ISOTROPIC);
    box.scaleBox(diagonalMatrix(linalg::Vec3D{2.0, 2.0, 2.0}));
    EXPECT_VECTOR_NEAR(
        box.get_boxDimensions(),
        linalg::Vec3D(8.0, 10.0, 12.0),
        1.0e-12
    );
    EXPECT_NEAR(box.getVolume(), box.getVolume(), 1.0e-12);

    settings::ManostatSettings::setIsotropy(Isotropy::FULL_ANISOTROPIC);
    const auto originalAngles = box.getBoxAngles();
    box.scaleBox(diagonalMatrix(linalg::Vec3D{0.5, 0.5, 0.5}));
    EXPECT_VECTOR_NEAR(
        box.get_boxDimensions(),
        linalg::Vec3D(4.0, 5.0, 6.0),
        1.0e-12
    );
    EXPECT_VECTOR_NEAR(box.getBoxAngles(), originalAngles, 1.0e-12);

    auto [dimensions, angles] = molsys::calcBoxDimAndAnglesFromBoxMatrix(
        TestTriclinicBox::getBoxMatrix(box)
    );
    EXPECT_VECTOR_NEAR(dimensions, box.get_boxDimensions(), 1.0e-12);
    EXPECT_VECTOR_NEAR(
        angles.numerical_value_in(mstd::units::deg),
        box.getBoxAngles(),
        1.0e-12
    );
}
