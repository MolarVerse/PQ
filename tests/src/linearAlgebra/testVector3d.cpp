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

#include <gtest/gtest.h>   // for Test, EXPECT_EQ, TestInfo (ptr only), ASSER...

#include <array>      // for array
#include <cmath>      // for sqrt
#include <iostream>   // for cout, ostream
#include <string>     // for allocator, string

#include "gtest/gtest.h"   // for Message, TestPartResult, AssertionResult
#include "vector3d.hpp"    // for Vec3D, Vec3Di, Vec3Dul, maximum, minimum

using namespace linearAlgebra;

/**
 * @file testVector3d.cpp
 *
 * @brief Contains tests for double, int and size_t Vector3D
 */

/**
 * @brief tests constructors for Vector3D
 *
 */
TEST(TestVector3d, constructors)
{
    auto vec = Vec3D(0.0, 1.0, 2.0);
    ASSERT_EQ(vec[0], 0.0);
    ASSERT_EQ(vec[1], 1.0);
    ASSERT_EQ(vec[2], 2.0);

    const auto vec2 = Vec3D(vec);
    ASSERT_EQ(vec2, vec);

    vec = Vec3D(1.0);
    ASSERT_EQ(vec[0], 1.0);
    ASSERT_EQ(vec[1], 1.0);
    ASSERT_EQ(vec[2], 1.0);

    auto vecInt = Vec3Di(0, 1, 2);
    ASSERT_EQ(vecInt[0], 0);
    ASSERT_EQ(vecInt[1], 1);
    ASSERT_EQ(vecInt[2], 2);

    const auto vec2i = Vec3Di(vecInt);
    ASSERT_EQ(vec2i, vecInt);

    vecInt = Vec3Di(1);
    ASSERT_EQ(vecInt[0], 1);
    ASSERT_EQ(vecInt[1], 1);
    ASSERT_EQ(vecInt[2], 1);

    auto vecUnsignedLong = Vec3Dul(0, 1, 2);
    ASSERT_EQ(vecUnsignedLong[0], 0);
    ASSERT_EQ(vecUnsignedLong[1], 1);
    ASSERT_EQ(vecUnsignedLong[2], 2);

    const auto vec2ul = Vec3Dul(vecUnsignedLong);
    ASSERT_EQ(vec2ul, vecUnsignedLong);

    vecUnsignedLong = Vec3Dul(1);
    ASSERT_EQ(vecUnsignedLong[0], 1);
    ASSERT_EQ(vecUnsignedLong[1], 1);
    ASSERT_EQ(vecUnsignedLong[2], 1);

    auto vectorVec3D = Vector3D<Vec3D>{vec, vec2, vec};
    ASSERT_EQ(vectorVec3D[0], vec);
    ASSERT_EQ(vectorVec3D[1], vec2);
    ASSERT_EQ(vectorVec3D[2], vec);

    const auto vectorVec3D2 = Vector3D<Vec3D>(vectorVec3D);
    ASSERT_EQ(vectorVec3D2, vectorVec3D);

    vectorVec3D = Vector3D<Vec3D>(vec);
    ASSERT_EQ(vectorVec3D[0], vec);
    ASSERT_EQ(vectorVec3D[1], vec);
    ASSERT_EQ(vectorVec3D[2], vec);
}

/**
 * @brief tests assignment operators for Vector3D
 *
 */
TEST(TestVector3d, assignmentOperator)
{
    Vec3D vec1(0.0, 1.0, 2.0);
    Vec3D vec2(1.0, 2.0, 3.0);

    vec1 = vec2;
    ASSERT_EQ(vec1, vec2);

    Vec3Di vec3(0, 1, 2);
    Vec3Di vec4(1, 2, 3);

    vec3 = vec4;
    ASSERT_EQ(vec3, vec4);

    Vec3Dul vec5(0, 1, 2);
    Vec3Dul vec6(1, 2, 3);

    vec5 = vec6;
    ASSERT_EQ(vec5, vec6);

    const Vec3D           vec7(0.0, 1.0, 2.0);
    const Vec3D           vec8(1.0, 2.0, 3.0);
    Vector3D<Vec3D>       vec9(vec7, vec7, vec7);
    const Vector3D<Vec3D> vec10(vec8, vec8, vec8);

    vec9 = vec10;
    ASSERT_EQ(vec9, vec10);
}

/**
 * @brief tests operator+ for Vector3D
 *
 */
TEST(TestVector3d, additionOperator)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);
    const auto vec2 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(vec1 + vec2, Vec3D(1.0, 3.0, 5.0));
    EXPECT_EQ(vec1 + 1.0, Vec3D(1.0, 2.0, 3.0));

    const auto vec3 = Vec3Di(1, 2, 3);
    const auto vec4 = Vec3Di(2, 3, 4);

    EXPECT_EQ(vec3 + vec4, Vec3Di(3, 5, 7));
    EXPECT_EQ(vec3 + 1, Vec3Di(2, 3, 4));

    const auto vec5 = Vec3Dul(1, 2, 3);
    const auto vec6 = Vec3Dul(2, 3, 4);

    EXPECT_EQ(vec5 + vec6, Vec3Dul(3, 5, 7));
    EXPECT_EQ(vec5 + 1, Vec3Dul(2, 3, 4));

    const auto vec7 = Vector3D<Vec3D>{vec1, vec2, vec1};
    const auto vec8 = Vector3D<Vec3D>{vec2, vec1, vec2};

    EXPECT_EQ(
        vec7 + vec8,
        Vector3D<Vec3D>(vec1 + vec2, vec2 + vec1, vec1 + vec2)
    );
    EXPECT_EQ(
        vec7 + vec1,
        Vector3D<Vec3D>(vec1 + vec1, vec2 + vec1, vec1 + vec1)
    );
}

/**
 * @brief tests operator- for Vector3D
 *
 */
TEST(TestVector3d, subtractionOperator)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);
    const auto vec2 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(vec1 - vec2, Vec3D(-1.0, -1.0, -1.0));
    EXPECT_EQ(vec1 - 1.0, Vec3D(-1.0, 0.0, 1.0));
    EXPECT_EQ(-vec1, Vec3D(0.0, -1.0, -2.0));

    const auto vec3 = Vec3Di(1, 2, 3);
    const auto vec4 = Vec3Di(2, 3, 4);

    EXPECT_EQ(vec3 - vec4, Vec3Di(-1, -1, -1));
    EXPECT_EQ(vec3 - 1, Vec3Di(0, 1, 2));
    EXPECT_EQ(-vec3, Vec3Di(-1, -2, -3));

    const auto vec5 = Vec3Dul(1, 2, 3);
    const auto vec6 = Vec3Dul(2, 3, 4);

    EXPECT_EQ(vec6 - vec5, Vec3Dul(1, 1, 1));
    EXPECT_EQ(vec6 - 1, Vec3Dul(1, 2, 3));
    EXPECT_EQ(-vec5, Vec3Dul(-1, -2, -3));

    const auto vec7 = Vector3D<Vec3D>{vec1, vec2, vec1};
    const auto vec8 = Vector3D<Vec3D>{vec2, vec1, vec2};

    EXPECT_EQ(
        vec7 - vec8,
        Vector3D<Vec3D>(vec1 - vec2, vec2 - vec1, vec1 - vec2)
    );
    EXPECT_EQ(
        vec7 - vec1,
        Vector3D<Vec3D>(vec1 - vec1, vec2 - vec1, vec1 - vec1)
    );
    EXPECT_EQ(-vec7, Vector3D<Vec3D>(-vec1, -vec2, -vec1));
}

/**
 * @brief tests operator* for Vector3D
 *
 */
TEST(TestVector3d, multiplicationOperator)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);
    const auto vec2 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(vec1 * vec2, Vec3D(0.0, 2.0, 6.0));
    EXPECT_EQ(vec1 * 2.0, Vec3D(0.0, 2.0, 4.0));
    EXPECT_EQ(2.0 * vec1, Vec3D(0.0, 2.0, 4.0));

    const auto vec3 = Vec3Di(1, 2, 3);
    const auto vec4 = Vec3Di(2, 3, 4);

    EXPECT_EQ(vec3 * vec4, Vec3Di(2, 6, 12));
    EXPECT_EQ(vec3 * 2, Vec3Di(2, 4, 6));

    const auto vec5 = Vec3Dul(1, 2, 3);
    const auto vec6 = Vec3Dul(2, 3, 4);

    EXPECT_EQ(vec5 * vec6, Vec3Dul(2, 6, 12));
    EXPECT_EQ(vec5 * 2, Vec3Dul(2, 4, 6));

    const auto vec7 = Vector3D<Vec3D>{vec1, vec2, vec1};
    const auto vec8 = Vector3D<Vec3D>{vec2, vec1, vec2};

    EXPECT_EQ(
        vec7 * vec8,
        Vector3D<Vec3D>(vec1 * vec2, vec2 * vec1, vec1 * vec2)
    );
    EXPECT_EQ(
        vec7 * vec1,
        Vector3D<Vec3D>(vec1 * vec1, vec2 * vec1, vec1 * vec1)
    );
}

/**
 * @brief tests operator/ for Vector3D
 *
 */
TEST(TestVector3d, divisionOperator)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);
    const auto vec2 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(vec1 / vec2, Vec3D(0.0, 0.5, 2.0 / 3.0));
    EXPECT_EQ(vec1 / 2.0, Vec3D(0.0, 0.5, 1.0));
    EXPECT_EQ(2.0 / vec2, Vec3D(2.0, 1.0, 2.0 / 3.0));

    const auto vec3 = Vec3Di(1, 2, 3);
    const auto vec4 = Vec3Di(2, 3, 4);

    EXPECT_EQ(vec3 / vec4, Vec3Di(0, 0, 0));
    EXPECT_EQ(vec3 / 2, Vec3Di(0, 1, 1));

    const auto vec5 = Vec3Dul(1, 2, 3);
    const auto vec6 = Vec3Dul(2, 3, 4);

    EXPECT_EQ(vec5 / vec6, Vec3Dul(0, 0, 0));
    EXPECT_EQ(vec5 / 2, Vec3Dul(0, 1, 1));

    const auto vec7 = Vector3D<Vec3D>{vec1, vec2, vec1};
    const auto vec8 = Vector3D<Vec3D>{vec2, vec1, vec2};

    EXPECT_EQ(
        vec7 / vec8,
        Vector3D<Vec3D>(vec1 / vec2, vec2 / vec1, vec1 / vec2)
    );
    EXPECT_EQ(
        vec7 / vec2,
        Vector3D<Vec3D>(vec1 / vec2, vec2 / vec2, vec1 / vec2)
    );
}

/**
 * @brief tests operator< for Vector3D
 *
 */
TEST(TestVector3d, isLessOperator)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_TRUE(vec1 < 3.0);
    EXPECT_FALSE(vec1 < 2.0);

    const auto vec2 = Vec3Di(0, 1, 2);

    EXPECT_TRUE(vec2 < 3);
    EXPECT_FALSE(vec2 < 2);

    const auto vec3 = Vec3Dul(0, 1, 2);

    EXPECT_TRUE(vec3 < 3);
    EXPECT_FALSE(vec3 < 2);
}

/**
 * @brief tests operator> for Vector3D
 *
 */
TEST(TestVector3d, isGreaterOperator)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_TRUE(vec1 > -1.0);
    EXPECT_FALSE(vec1 > 1.0);

    const auto vec2 = Vec3Di(0, 1, 2);

    EXPECT_TRUE(vec2 > -1);
    EXPECT_FALSE(vec2 > 1);

    const auto vec3 = Vec3Dul(1, 2, 3);

    EXPECT_TRUE(vec3 > 0);
    EXPECT_FALSE(vec3 > 1);
}

/**
 * @brief tests fabs for Vector3D
 *
 */
TEST(TestVector3d, fABS)
{
    const auto vec1 = Vec3D(0.0, -1.0, 2.0);
    EXPECT_EQ(fabs(vec1), Vec3D(0.0, 1.0, 2.0));
}

/**
 * @brief tests static casts for Vector3D
 *
 */
TEST(TestVector3d, staticCast)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);
    const auto vec2 = Vec3Di(0, 1, 2);
    const auto vec3 = Vec3Dul(0, 1, 2);

    EXPECT_EQ(static_cast<Vec3Di>(vec1), vec2);
    EXPECT_EQ(static_cast<Vec3Dul>(vec1), vec3);
    EXPECT_EQ(static_cast<Vec3D>(vec2), vec1);
    EXPECT_EQ(static_cast<Vec3Dul>(vec2), vec3);
    EXPECT_EQ(static_cast<Vec3D>(vec3), vec1);
    EXPECT_EQ(static_cast<Vec3Di>(vec3), vec2);
}

/**
 * @brief tests round for Vector3D
 *
 */
TEST(TestVector3d, round)
{
    const auto vec = Vec3D(0.0, 1.4, 2.8);

    EXPECT_EQ(round(vec), Vec3D(0.0, 1.0, 3.0));
}

/**
 * @brief tests ceil for Vector3D
 *
 */
TEST(TestVector3d, ceil)
{
    const auto vec = Vec3D(0.0, 1.4, 2.8);

    EXPECT_EQ(ceil(vec), Vec3D(0.0, 2.0, 3.0));
}

/**
 * @brief tests floor for Vector3D
 *
 */
TEST(TestVector3d, floor)
{
    const auto vec = Vec3D(0.0, 1.4, 2.8);

    EXPECT_EQ(floor(vec), Vec3D(0.0, 1.0, 2.0));
}

/**
 * @brief tests abs for Vector3D
 *
 */
TEST(TestVector3d, norm)
{
    const auto vec = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(norm(vec), sqrt(5.0));
}

/**
 * @brief tests norm squared for Vector3D
 *
 */
TEST(TestVector3d, normSquared)
{
    const auto vec = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(normSquared(vec), 5.0);
}

/**
 * @brief tests minimum for Vector3D
 *
 */
TEST(TestVector3d, minimum)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(minimum(vec1), 0.0);

    const auto vec2 = Vec3Di(0, 1, 2);

    EXPECT_EQ(minimum(vec2), 0);

    const auto vec3 = Vec3Dul(0, 1, 2);

    EXPECT_EQ(minimum(vec3), 0);
}

/**
 * @brief tests maximum for Vector3D
 *
 */
TEST(TestVector3d, maximum)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(maximum(vec1), 2.0);

    const auto vec2 = Vec3Di(0, 1, 2);

    EXPECT_EQ(maximum(vec2), 2);

    const auto vec3 = Vec3Dul(0, 1, 2);

    EXPECT_EQ(maximum(vec3), 2);
}

/**
 * @brief tests sum for Vector3D
 *
 */
TEST(TestVector3d, sum)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(sum(vec1), 3.0);

    const auto vec2 = Vec3Di(0, 1, 2);

    EXPECT_EQ(sum(vec2), 3);

    const auto vec3 = Vec3Dul(0, 1, 2);

    EXPECT_EQ(sum(vec3), 3);
}

/**
 * @brief tests mean for Vector3D
 *
 */
TEST(TestVector3d, mean)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(mean(vec1), 1.0);
}

/**
 * @brief tests product of entries for Vector3D
 *
 */
TEST(TestVector3d, product)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(prod(vec1), 6.0);

    const auto vec2 = Vec3Di(1, 2, 3);

    EXPECT_EQ(prod(vec2), 6);

    const auto vec3 = Vec3Dul(1, 2, 3);

    EXPECT_EQ(prod(vec3), 6);
}

/**
 * @brief tests dot product for Vector3D
 *
 */
TEST(TestVector3d, scalarProduct)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);
    const auto vec2 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(dot(vec1, vec2), 14.0);
}

/**
 * @brief tests cross product for Vector3D
 *
 */
TEST(TestVector3d, crossProduct)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);
    const auto vec2 = Vec3D(2.0, 3.0, 4.0);

    EXPECT_EQ(cross(vec1, vec2), Vec3D(-1.0, 2.0, -1.0));
}

/**
 * @brief tests cos for Vector3D
 *
 */
TEST(TestVector3D, cos)
{
    const auto vec1 = Vec3D(0.0, 90.0, 180.0);

    auto result = cos(vec1 * M_PI / 180.0);

    EXPECT_NEAR(result[0], 1.0, 1e-15);
    EXPECT_NEAR(result[1], 0.0, 1e-15);
    EXPECT_NEAR(result[2], -1.0, 1e-15);
}

/**
 * @brief tests cos for two Vector3D objects
 *
 * @details calculates the cosine of the angle between two vectors
 *
 */
TEST(TestVector3d, cosBetweenVectors)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);
    const auto vec2 = Vec3D(2.0, 3.0, 4.0);

    EXPECT_DOUBLE_EQ(cos(vec1, vec2), 0.9925833339709303);
}

/**
 * @brief tests angle for Vector3D
 *
 * @details calculates the angle between two vectors
 *
 */
TEST(TestVector3d, angle)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);
    const auto vec2 = Vec3D(2.0, 3.0, 4.0);

    EXPECT_DOUBLE_EQ(angle(vec1, vec2), ::acos(cos(vec1, vec2)));
}

/**
 * @brief tests os stream for Vector3D
 *
 */
TEST(TestVector3d, osStream)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);
    testing::internal::CaptureStdout();
    std::cout << vec1;
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "1 2 3");

    const auto vec2 = Vec3Di(1, 2, 3);
    testing::internal::CaptureStdout();
    std::cout << vec2;
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "1 2 3");

    const auto vec3 = Vec3Dul(1, 2, 3);
    testing::internal::CaptureStdout();
    std::cout << vec3;
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "1 2 3");
}

/**
 * @brief tests begin() for Vector3D
 *
 */
TEST(TestVector3d, begin)
{
    const auto vec = Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(*vec.begin(), 1.0);

    const auto vec2 = Vec3Di(1, 2, 3);
    EXPECT_EQ(*vec2.begin(), 1);

    const auto vec3 = Vec3Dul(1, 2, 3);
    EXPECT_EQ(*vec3.begin(), 1);

    auto vec4 = Vector3D<Vec3D>{vec, vec + 1, vec + 2};
    EXPECT_EQ(*vec4.begin(), vec);
}

/**
 * @brief tests end() for Vector3D
 *
 */
TEST(TestVector3d, end)
{
    const auto vec = Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(*(vec.end() - 1), 3.0);

    const auto vec2 = Vec3Di(1, 2, 3);
    EXPECT_EQ(*(vec2.end() - 1), 3);

    const auto vec3 = Vec3Dul(1, 2, 3);
    EXPECT_EQ(*(vec3.end() - 1), 3);

    auto vec4 = Vector3D<Vec3D>{vec, vec + 1, vec + 2};
    EXPECT_EQ(*(vec4.end() - 1), vec + 2);
}

/**
 * @brief tests operator== for Vector3D
 *
 */
TEST(TestVector3d, equalOperator)
{
    auto  vec1 = Vec3D(1.0, 2.0, 3.0);
    Vec3D vec2 = vec1;

    EXPECT_EQ(vec1, vec2);

    auto   vec3 = Vec3Di(1, 2, 3);
    Vec3Di vec4 = vec3;

    EXPECT_EQ(vec3, vec4);

    auto    vec5 = Vec3Dul(1, 2, 3);
    Vec3Dul vec6 = vec5;

    EXPECT_EQ(vec5, vec6);
}

/**
 * @brief tests addition assignment operator for Vector3D
 *
 */
TEST(TestVector3d, additionAssignmentOperator)
{
    auto vec  = Vec3D(1.0, 2.0, 3.0);
    vec      += Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(vec, Vec3D(2.0, 4.0, 6.0));
    vec += 1.0;
    EXPECT_EQ(vec, Vec3D(3.0, 5.0, 7.0));

    auto vec2  = Vec3Di(1, 2, 3);
    vec2      += Vec3Di(1, 2, 3);
    EXPECT_EQ(vec2, Vec3Di(2, 4, 6));
    vec2 += 1;
    EXPECT_EQ(vec2, Vec3Di(3, 5, 7));

    auto vec3  = Vec3Dul(1, 2, 3);
    vec3      += Vec3Dul(1, 2, 3);
    EXPECT_EQ(vec3, Vec3Dul(2, 4, 6));
    vec3 += 1;
    EXPECT_EQ(vec3, Vec3Dul(3, 5, 7));

    auto subVec  = Vec3D(1.0, 1.0, 1.0);
    auto vec4    = Vector3D<Vec3D>{vec, subVec, vec};
    vec4        += vec4;
    EXPECT_EQ(vec4, Vector3D<Vec3D>(vec + vec, subVec + subVec, vec + vec));
    vec4 += vec;
    EXPECT_EQ(
        vec4,
        Vector3D<Vec3D>(vec + vec + vec, subVec + subVec + vec, vec + vec + vec)
    );
}

/**
 * @brief tests subtraction assignment operator for Vector3D
 *
 */
TEST(TestVector3d, subtractionAssignmentOperator)
{
    auto vec  = Vec3D(1.0, 2.0, 3.0);
    vec      -= Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(vec, Vec3D(0.0, 0.0, 0.0));
    vec -= 1.0;
    EXPECT_EQ(vec, Vec3D(-1.0, -1.0, -1.0));

    auto vec2  = Vec3Di(1, 2, 3);
    vec2      -= Vec3Di(1, 2, 3);
    EXPECT_EQ(vec2, Vec3Di(0, 0, 0));
    vec2 -= 1;
    EXPECT_EQ(vec2, Vec3Di(-1, -1, -1));

    auto vec3  = Vec3Dul(1, 2, 3);
    vec3      -= Vec3Dul(1, 2, 3);
    EXPECT_EQ(vec3, Vec3Dul(0, 0, 0));
    vec3 -= 1;
    EXPECT_EQ(vec3, Vec3Dul(-1, -1, -1));

    auto subVec  = Vec3D(1.0, 1.0, 1.0);
    auto vec4    = Vector3D<Vec3D>{vec, subVec, vec};
    vec4        -= vec4;
    EXPECT_EQ(vec4, Vector3D<Vec3D>(vec - vec, subVec - subVec, vec - vec));
    vec4 -= vec;
    EXPECT_EQ(
        vec4,
        Vector3D<Vec3D>(vec - vec - vec, subVec - subVec - vec, vec - vec - vec)
    );
}

/**
 * @brief tests multiplication assignment operator for Vector3D
 *
 */
TEST(TestVector3d, multiplicationAssignmentOperator)
{
    auto vec  = Vec3D(1.0, 2.0, 3.0);
    vec      *= Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(vec, Vec3D(1.0, 4.0, 9.0));
    vec *= 2.0;
    EXPECT_EQ(vec, Vec3D(2.0, 8.0, 18.0));

    auto vec2  = Vec3Di(1, 2, 3);
    vec2      *= Vec3Di(1, 2, 3);
    EXPECT_EQ(vec2, Vec3Di(1, 4, 9));
    vec2 *= 2;
    EXPECT_EQ(vec2, Vec3Di(2, 8, 18));

    auto vec3  = Vec3Dul(1, 2, 3);
    vec3      *= Vec3Dul(1, 2, 3);
    EXPECT_EQ(vec3, Vec3Dul(1, 4, 9));
    vec3 *= 2;
    EXPECT_EQ(vec3, Vec3Dul(2, 8, 18));

    auto subVec  = Vec3D(1.0, 1.0, 1.0);
    auto vec4    = Vector3D<Vec3D>{vec, subVec, vec};
    vec4        *= vec4;
    EXPECT_EQ(vec4, Vector3D<Vec3D>(vec * vec, subVec * subVec, vec * vec));
    vec4 *= vec;
    EXPECT_EQ(
        vec4,
        Vector3D<Vec3D>(vec * vec * vec, subVec * subVec * vec, vec * vec * vec)
    );
}

/**
 * @brief tests division assignment operator for Vector3D
 *
 */
TEST(TestVector3d, divisionAssignmentOperator)
{
    auto vec  = Vec3D(1.0, 2.0, 3.0);
    vec      /= Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(vec, Vec3D(1.0, 1.0, 1.0));
    vec /= 2.0;
    EXPECT_EQ(vec, Vec3D(0.5, 0.5, 0.5));

    auto vec2  = Vec3Di(1, 2, 3);
    vec2      /= Vec3Di(1, 2, 3);
    EXPECT_EQ(vec2, Vec3Di(1, 1, 1));
    vec2 /= 2;
    EXPECT_EQ(vec2, Vec3Di(0, 0, 0));

    auto vec3  = Vec3Dul(1, 2, 3);
    vec3      /= Vec3Dul(1, 2, 3);
    EXPECT_EQ(vec3, Vec3Dul(1, 1, 1));
    vec3 /= 2;
    EXPECT_EQ(vec3, Vec3Dul(0, 0, 0));

    auto subVec  = Vec3D(1.0, 1.0, 1.0);
    auto vec4    = Vector3D<Vec3D>{vec, subVec, vec};
    vec4        /= vec4;
    EXPECT_EQ(vec4, Vector3D<Vec3D>(vec / vec, subVec / subVec, vec / vec));
    vec4 /= vec;
    EXPECT_EQ(
        vec4,
        Vector3D<Vec3D>(vec / vec / vec, subVec / subVec / vec, vec / vec / vec)
    );
}