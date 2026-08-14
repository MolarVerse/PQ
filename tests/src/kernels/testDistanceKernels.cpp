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
#include "gtest/gtest.h"
#include "simulationBox.hpp"

using namespace kernel;

TEST(TestDistanceKernels, distVecNoPBCIsSimpleSubtraction)
{
    const auto a = linearAlgebra::Vec3D(2.0, 3.0, 4.0);
    const auto b = linearAlgebra::Vec3D(1.0, 1.0, 1.0);

    EXPECT_EQ(distVec(a, b), linearAlgebra::Vec3D(1.0, 2.0, 3.0));
    EXPECT_EQ(distVec(a, a), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    EXPECT_EQ(distVec(b, a), linearAlgebra::Vec3D(-1.0, -2.0, -3.0));
}

TEST(TestDistanceKernels, distVecAndDist2NoPBCMatchesAnalyticalDistanceSquared)
{
    const auto a = linearAlgebra::Vec3D(1.0, 2.0, 2.0);
    const auto b = linearAlgebra::Vec3D(0.0, 0.0, 0.0);

    const auto [dxyz, r2] = distVecAndDist2(a, b);
    EXPECT_EQ(dxyz, linearAlgebra::Vec3D(1.0, 2.0, 2.0));
    EXPECT_DOUBLE_EQ(r2, 1.0 + 4.0 + 4.0);
}

TEST(TestDistanceKernels, distVecWithPBCChoosesMinimumImage)
{
    // 10 x 10 x 10 orthorhombic box. Two atoms at (0.5, 0, 0) and
    // (9.5, 0, 0) should be 1.0 apart under minimum image, not 9.0.
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    const auto a   = linearAlgebra::Vec3D(0.5, 0.0, 0.0);
    const auto b   = linearAlgebra::Vec3D(9.5, 0.0, 0.0);
    const auto dxy = distVec(a, b, box);

    EXPECT_NEAR(linearAlgebra::norm(dxy), 1.0, 1e-12);
}

TEST(TestDistanceKernels, distVecAndDist2WithPBCConsistentWithDistVec)
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({8.0, 8.0, 8.0});

    const auto a = linearAlgebra::Vec3D(0.0, 0.0, 0.0);
    const auto b = linearAlgebra::Vec3D(3.0, 4.0, 0.0);

    const auto dxyzOnly   = distVec(a, b, box);
    const auto [dxyz, r2] = distVecAndDist2(a, b, box);
    EXPECT_EQ(dxyzOnly, dxyz);
    EXPECT_DOUBLE_EQ(r2, linearAlgebra::normSquared(dxyz));
}

TEST(TestDistanceKernels, distVecWithPBCIsSymmetricAcrossAllAxes)
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 12.0, 14.0});

    const auto a = linearAlgebra::Vec3D(4.8, -5.5, 6.2);
    const auto b = linearAlgebra::Vec3D(-4.7, 5.6, -6.1);

    const auto ab        = distVec(a, b, box);
    const auto ba        = distVec(b, a, box);
    const auto [ab2, r2] = distVecAndDist2(a, b, box);

    EXPECT_NEAR(ab[0], -0.5, 1e-12);
    EXPECT_NEAR(ab[1], 0.9, 1e-12);
    EXPECT_NEAR(ab[2], -1.7, 1e-12);

    EXPECT_EQ(ab, ab2);
    EXPECT_NEAR(r2, linearAlgebra::normSquared(ab), 1e-12);
    EXPECT_NEAR(distSquared(a, b, box), distSquared(b, a, box), 1e-12);

    EXPECT_NEAR(ab[0], -ba[0], 1e-12);
    EXPECT_NEAR(ab[1], -ba[1], 1e-12);
    EXPECT_NEAR(ab[2], -ba[2], 1e-12);
}

TEST(TestDistanceKernels, distSquaredWithPBCMinimumImageDistance)
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    const auto a = linearAlgebra::Vec3D(0.5, 0.0, 0.0);
    const auto b = linearAlgebra::Vec3D(9.5, 0.0, 0.0);

    EXPECT_NEAR(distSquared(a, b, box), 1.0, 1e-12);
}

TEST(TestDistanceKernels, distVecZeroInputs)
{
    const auto a = linearAlgebra::Vec3D(0.0, 0.0, 0.0);

    EXPECT_EQ(distVec(a, a), linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    const auto [dxyz, r2] = distVecAndDist2(a, a);
    EXPECT_EQ(dxyz, linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(r2, 0.0);
}
