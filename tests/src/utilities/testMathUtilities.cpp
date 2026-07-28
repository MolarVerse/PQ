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

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), EXPECT_EQ

#include <limits>   // for numeric_limits
#include <string>   // for allocator, string

#include "gtest/gtest.h"       // for AssertionResult, Message, TestPartResult
#include "mathUtilities.hpp"   // for compare, sign, utilities
#include "vector3d.hpp"        // IWYU pragma: keep - for Vec3D

using namespace utilities;

/**
 * @brief tests compare function for double type
 *
 */
TEST(TestMathUtilities, compare)
{
    const double a = 1.0;
    EXPECT_TRUE(compare(a, a));
    EXPECT_FALSE(compare(a, a + std::numeric_limits<double>::epsilon()));

    const auto &b = linearAlgebra::Vec3D(1.0, 2.0, 3.0);
    EXPECT_TRUE(compare(b, b));
    EXPECT_FALSE(compare(
        b,
        b + linearAlgebra::Vec3D(
                b[0],
                b[1],
                std::numeric_limits<double>::epsilon()
            )
    ));
}

/**
 * @brief tests sign template function (here tests only for double data type)
 *
 */
TEST(TestMathUtilities, sign)
{
    EXPECT_EQ(sign(2.0), 1);
    EXPECT_EQ(sign(-2.0), -1);
    EXPECT_EQ(sign(0.0), 0);
}

/**
 * @brief tests compare<T>(a, b, tolerance) — 3-arg overload with a
 * user-supplied tolerance.
 */
TEST(TestMathUtilities, compareWithTolerance)
{
    // compare uses strict `<`, so a == b only compares equal when the
    // tolerance is strictly positive.
    EXPECT_TRUE(compare(1.0, 1.0 + 1e-9, 1e-8));
    EXPECT_FALSE(compare(1.0, 1.0 + 1e-7, 1e-8));
    EXPECT_FALSE(compare(0.0, 0.0, 0.0));
    EXPECT_TRUE(compare(0.0, 0.0, 1e-12));
    EXPECT_FALSE(compare(1.0, 2.0, 0.5));
}

/**
 * @brief tests compare(Vec3D, Vec3D, tolerance) — Vec3D compare with
 * a user-supplied tolerance.
 */
TEST(TestMathUtilities, compareVec3DWithTolerance)
{
    const auto a = linearAlgebra::Vec3D(1.0, 2.0, 3.0);
    const auto b = linearAlgebra::Vec3D(1.0 + 1e-9, 2.0, 3.0 - 1e-9);
    EXPECT_TRUE(compare(a, b, 1e-8));
    EXPECT_FALSE(compare(a, b, 1e-10));
}

/**
 * @brief tests kroneckerDelta(i, j): 1 when i == j, 0 otherwise.
 */
TEST(TestMathUtilities, kroneckerDelta)
{
    EXPECT_EQ(kroneckerDelta(0u, 0u), 1u);
    EXPECT_EQ(kroneckerDelta(1u, 1u), 1u);
    EXPECT_EQ(kroneckerDelta(0u, 1u), 0u);
    EXPECT_EQ(kroneckerDelta(5u, 7u), 0u);
}

/**
 * @brief tests isZero template function for the double data type. Uses
 * exact equality, so subnormal but non-zero values are not "zero".
 */
TEST(TestMathUtilities, isZero)
{
    EXPECT_TRUE(isZero(0.0));
    EXPECT_TRUE(isZero(-0.0));
    EXPECT_FALSE(isZero(1.0));
    EXPECT_FALSE(isZero(std::numeric_limits<double>::epsilon()));
    EXPECT_FALSE(isZero(std::numeric_limits<double>::min()));
}