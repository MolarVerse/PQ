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
#include "simulationBox.hpp"

using namespace kernel;

TEST(TestDistanceKernels, distVecNoPBCIsSimpleSubtraction)
{
    const auto vec1 = linalg::Vec3D(2.0, 3.0, 4.0);
    const auto vec2 = linalg::Vec3D(1.0, 1.0, 1.0);

    EXPECT_EQ(distVec(vec1, vec2), linalg::Vec3D(1.0, 2.0, 3.0));
    EXPECT_EQ(distVec(vec1, vec1), linalg::Vec3D(0.0, 0.0, 0.0));
    EXPECT_EQ(distVec(vec2, vec1), linalg::Vec3D(-1.0, -2.0, -3.0));
}

TEST(TestDistanceKernels, distVecAndDist2NoPBCMatchesAnalyticalDistanceSquared)
{
    const auto vec1 = linalg::Vec3D(1.0, 2.0, 2.0);
    const auto vec2 = linalg::Vec3D(0.0, 0.0, 0.0);

    const auto [dxyz, rSquared] = distVecAndDist2(vec1, vec2);
    EXPECT_EQ(dxyz, linalg::Vec3D(1.0, 2.0, 2.0));
    EXPECT_DOUBLE_EQ(rSquared, 1.0 + 4.0 + 4.0);
}

TEST(TestDistanceKernels, distVecWithPBCChoosesMinimumImage)
{
    // 10 x 10 x 10 orthorhombic box. Two atoms at (0.5, 0, 0) and
    // (9.5, 0, 0) should be 1.0 apart under minimum image, not 9.0.
    auto box = molsys::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    const auto vec1 = linalg::Vec3D(0.5, 0.0, 0.0);
    const auto vec2 = linalg::Vec3D(9.5, 0.0, 0.0);
    const auto dxy  = distVec(vec1, vec2, box);

    EXPECT_NEAR(linalg::norm(dxy), 1.0, 1e-12);
}

TEST(TestDistanceKernels, distVecAndDist2WithPBCConsistentWithDistVec)
{
    auto box = molsys::SimulationBox();
    box.setBoxDimensions({8.0, 8.0, 8.0});

    const auto vec1 = linalg::Vec3D(0.0, 0.0, 0.0);
    const auto vec2 = linalg::Vec3D(3.0, 4.0, 0.0);

    const auto dxyzOnly         = distVec(vec1, vec2, box);
    const auto [dxyz, rSquared] = distVecAndDist2(vec1, vec2, box);
    EXPECT_EQ(dxyzOnly, dxyz);
    EXPECT_DOUBLE_EQ(rSquared, linalg::normSquared(dxyz));
}

TEST(TestDistanceKernels, distVecWithPBCIsSymmetricAcrossAllAxes)
{
    auto box = molsys::SimulationBox();
    box.setBoxDimensions({10.0, 12.0, 14.0});

    const auto vec1 = linalg::Vec3D(4.8, -5.5, 6.2);
    const auto vec2 = linalg::Vec3D(-4.7, 5.6, -6.1);

    const auto vec12           = distVec(vec1, vec2, box);
    const auto vec21           = distVec(vec2, vec1, box);
    const auto [ab2, rSquared] = distVecAndDist2(vec1, vec2, box);

    EXPECT_NEAR(vec12[0], -0.5, 1e-12);
    EXPECT_NEAR(vec12[1], 0.9, 1e-12);
    EXPECT_NEAR(vec12[2], -1.7, 1e-12);

    EXPECT_EQ(vec12, ab2);
    EXPECT_NEAR(rSquared, linalg::normSquared(vec12), 1e-12);
    EXPECT_NEAR(
        distSquared(vec1, vec2, box),
        distSquared(vec2, vec1, box),
        1e-12
    );

    EXPECT_NEAR(vec12[0], -vec21[0], 1e-12);
    EXPECT_NEAR(vec12[1], -vec21[1], 1e-12);
    EXPECT_NEAR(vec12[2], -vec21[2], 1e-12);
}

TEST(TestDistanceKernels, distSquaredWithPBCMinimumImageDistance)
{
    auto box = molsys::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    const auto vec1 = linalg::Vec3D(0.5, 0.0, 0.0);
    const auto vec2 = linalg::Vec3D(9.5, 0.0, 0.0);

    EXPECT_NEAR(distSquared(vec1, vec2, box), 1.0, 1e-12);
}

TEST(TestDistanceKernels, distVecZeroInputs)
{
    const auto vec1 = linalg::Vec3D(0.0, 0.0, 0.0);

    EXPECT_EQ(distVec(vec1, vec1), linalg::Vec3D(0.0, 0.0, 0.0));

    const auto [dxyz, rSquared] = distVecAndDist2(vec1, vec1);
    EXPECT_EQ(dxyz, linalg::Vec3D(0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(rSquared, 0.0);
}
