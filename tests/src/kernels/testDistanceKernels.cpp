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

#include "distanceKernels.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"
#include "units.hpp"

TEST(TestDistanceKernels, distVecNoPBCIsSimpleSubtraction)
{
    const auto vec1 = linalg::Vec3D(2.0, 3.0, 4.0) * angstrom;
    const auto vec2 = linalg::Vec3D(1.0, 1.0, 1.0) * angstrom;

    EXPECT_EQ(
        kernel::distVec(vec1, vec2),
        linalg::Vec3D(1.0, 2.0, 3.0) * angstrom
    );
    EXPECT_EQ(
        kernel::distVec(vec1, vec1),
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom
    );
    EXPECT_EQ(
        kernel::distVec(vec2, vec1),
        linalg::Vec3D(-1.0, -2.0, -3.0) * angstrom
    );
}

TEST(TestDistanceKernels, distVecAndDist2NoPBCMatchesAnalyticalDistanceSquared)
{
    const auto vec1 = linalg::Vec3D(1.0, 2.0, 2.0) * angstrom;
    const auto vec2 = linalg::Vec3D(0.0, 0.0, 0.0) * angstrom;

    const auto [dxyz, rSquared] = kernel::distVecAndDist2(vec1, vec2);
    EXPECT_EQ(dxyz, linalg::Vec3D(1.0, 2.0, 2.0) * angstrom);
    EXPECT_DOUBLE_EQ(
        rSquared.numerical_value_in(angstrom * angstrom),
        1.0 + 4.0 + 4.0
    );
}

TEST(TestDistanceKernels, distVecWithPBCChoosesMinimumImage)
{
    // 10 x 10 x 10 orthorhombic box. Two atoms at (0.5, 0, 0) and
    // (9.5, 0, 0) should be 1.0 apart under minimum image, not 9.0.
    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    auto box = molsys::SimulationBox();
    box.setBox(orthoBox, false);

    const auto vec1 = linalg::Vec3D(0.5, 0.0, 0.0) * angstrom;
    const auto vec2 = linalg::Vec3D(9.5, 0.0, 0.0) * angstrom;
    const auto dxy  = kernel::distVec(vec1, vec2, box);

    EXPECT_NEAR(linalg::norm(dxy.numerical_value_in(angstrom)), 1.0, 1e-12);
}

TEST(TestDistanceKernels, distVecAndDist2WithPBCConsistentWithDistVec)
{
    auto                    box = molsys::SimulationBox();
    molsys::OrthorhombicBox orthoBox{
        {8.0 * angstrom, 8.0 * angstrom, 8.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    const auto vec1 = linalg::Vec3D(0.0, 0.0, 0.0) * angstrom;
    const auto vec2 = linalg::Vec3D(3.0, 4.0, 0.0) * angstrom;

    const auto dxyzOnly         = kernel::distVec(vec1, vec2, box);
    const auto [dxyz, rSquared] = kernel::distVecAndDist2(vec1, vec2, box);
    EXPECT_EQ(dxyzOnly, dxyz);
    EXPECT_DOUBLE_EQ(
        rSquared.numerical_value_in(angstrom * angstrom),
        linalg::normSquared(dxyz.numerical_value_in(angstrom))
    );
}

TEST(TestDistanceKernels, distVecWithPBCIsSymmetricAcrossAllAxes)
{
    auto                    box = molsys::SimulationBox();
    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 12.0 * angstrom, 14.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    const auto vec1 = linalg::Vec3D(4.8, -5.5, 6.2) * angstrom;
    const auto vec2 = linalg::Vec3D(-4.7, 5.6, -6.1) * angstrom;

    const auto vec12           = kernel::distVec(vec1, vec2, box);
    const auto vec21           = kernel::distVec(vec2, vec1, box);
    const auto [ab2, rSquared] = kernel::distVecAndDist2(vec1, vec2, box);

    EXPECT_NEAR(vec12[0].numerical_value_in(angstrom), -0.5, 1e-12);
    EXPECT_NEAR(vec12[1].numerical_value_in(angstrom), 0.9, 1e-12);
    EXPECT_NEAR(vec12[2].numerical_value_in(angstrom), -1.7, 1e-12);

    EXPECT_EQ(vec12, ab2);
    EXPECT_NEAR(
        rSquared.numerical_value_in(angstrom * angstrom),
        linalg::normSquared(vec12.numerical_value_in(angstrom)),
        1e-12
    );
    EXPECT_NEAR(
        kernel::distSquared(vec1, vec2, box)
            .numerical_value_in(angstrom * angstrom),
        kernel::distSquared(vec2, vec1, box)
            .numerical_value_in(angstrom * angstrom),
        1e-12
    );

    EXPECT_NEAR(
        vec12[0].numerical_value_in(angstrom),
        -vec21[0].numerical_value_in(angstrom),
        1e-12
    );
    EXPECT_NEAR(
        vec12[1].numerical_value_in(angstrom),
        -vec21[1].numerical_value_in(angstrom),
        1e-12
    );
    EXPECT_NEAR(
        vec12[2].numerical_value_in(angstrom),
        -vec21[2].numerical_value_in(angstrom),
        1e-12
    );
}

TEST(TestDistanceKernels, distSquaredWithPBCMinimumImageDistance)
{
    auto                    box = molsys::SimulationBox();
    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    const auto vec1 = linalg::Vec3D(0.5, 0.0, 0.0) * angstrom;
    const auto vec2 = linalg::Vec3D(9.5, 0.0, 0.0) * angstrom;

    EXPECT_NEAR(
        kernel::distSquared(vec1, vec2, box)
            .numerical_value_in(angstrom * angstrom),
        1.0,
        1e-12
    );
}

TEST(TestDistanceKernels, distVecZeroInputs)
{
    const auto vec1 = linalg::Vec3D(0.0, 0.0, 0.0) * angstrom;

    EXPECT_EQ(
        kernel::distVec(vec1, vec1),
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom
    );

    const auto [dxyz, rSquared] = kernel::distVecAndDist2(vec1, vec1);
    EXPECT_EQ(dxyz, Pos3D{Pos::zero()});
    EXPECT_DOUBLE_EQ(rSquared.numerical_value_in(angstrom * angstrom), 0.0);
}
