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

#include <limits>

#include "mathUtilities.hpp"
#include "vector3d.hpp"

/**
 * @brief tests utilities::compare function for double type
 *
 */
TEST(TestMathUtilities, compare)
{
    const double value1 = 1.0;
    EXPECT_TRUE(utilities::compare(value1, value1));
    EXPECT_FALSE(
        utilities::compare(
            value1,
            value1 + std::numeric_limits<double>::epsilon()
        )
    );

    const auto &value2 = linalg::Vec3D(1.0, 2.0, 3.0);
    EXPECT_TRUE(utilities::compare(value2, value2));
    EXPECT_FALSE(
        utilities::compare(
            value2,
            value2 + linalg::Vec3D(
                         value2[0],
                         value2[1],
                         std::numeric_limits<double>::epsilon()
                     )
        )
    );
}

/**
 * @brief tests sign template function (here tests only for double data type)
 *
 */
TEST(TestMathUtilities, sign)
{
    EXPECT_EQ(utilities::sign(2.0), 1);
    EXPECT_EQ(utilities::sign(-2.0), -1);
    EXPECT_EQ(utilities::sign(0.0), 0);
}

/**
 * @brief tests utilities::compare<T>(a, b, tolerance) — 3-arg overload with a
 * user-supplied tolerance.
 */
TEST(TestMathUtilities, compareWithTolerance)
{
    // utilities::compare uses strict `<`, so a == b only compares equal when
    // the tolerance is strictly positive.
    EXPECT_TRUE(utilities::compare(1.0, 1.0 + 1e-9, 1e-8));
    EXPECT_FALSE(utilities::compare(1.0, 1.0 + 1e-7, 1e-8));
    EXPECT_FALSE(utilities::compare(0.0, 0.0, 0.0));
    EXPECT_TRUE(utilities::compare(0.0, 0.0, 1e-12));
    EXPECT_FALSE(utilities::compare(1.0, 2.0, 0.5));
}

/**
 * @brief tests utilities::compare(Vec3D, Vec3D, tolerance) — Vec3D
 * utilities::compare with a user-supplied tolerance.
 */
TEST(TestMathUtilities, compareVec3DWithTolerance)
{
    const auto vec1 = linalg::Vec3D(1.0, 2.0, 3.0);
    const auto vec2 = linalg::Vec3D(1.0 + 1e-9, 2.0, 3.0 - 1e-9);
    EXPECT_TRUE(utilities::compare(vec1, vec2, 1e-8));
    EXPECT_FALSE(utilities::compare(vec1, vec2, 1e-10));
}

/**
 * @brief tests kroneckerDelta(i, j): 1 when i == j, 0 otherwise.
 */
TEST(TestMathUtilities, kroneckerDelta)
{
    EXPECT_EQ(utilities::kroneckerDelta(0U, 0U), 1U);
    EXPECT_EQ(utilities::kroneckerDelta(1U, 1U), 1U);
    EXPECT_EQ(utilities::kroneckerDelta(0U, 1U), 0U);
    EXPECT_EQ(utilities::kroneckerDelta(5U, 7U), 0U);
}

/**
 * @brief tests isZero template function for the double data type. Uses
 * exact equality, so subnormal but non-zero values are not "zero".
 */
TEST(TestMathUtilities, isZero)
{
    EXPECT_TRUE(utilities::isZero(0.0));
    EXPECT_TRUE(utilities::isZero(-0.0));
    EXPECT_FALSE(utilities::isZero(1.0));
    EXPECT_FALSE(utilities::isZero(std::numeric_limits<double>::epsilon()));
    EXPECT_FALSE(utilities::isZero(std::numeric_limits<double>::min()));
}
