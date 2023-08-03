#include "vector3d.hpp"

#include <gtest/gtest.h>

using namespace vector3d;

TEST(TestVector3d, testConstructors)
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

    const std::array arr = {0.0, 1.0, 2.0};
    vec                  = Vec3D(arr);
    ASSERT_EQ(vec[0], 0.0);
    ASSERT_EQ(vec[1], 1.0);
    ASSERT_EQ(vec[2], 2.0);

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

    const std::array<int, 3> arrInt = {0, 1, 2};
    vecInt                          = Vec3Di(arrInt);
    ASSERT_EQ(vecInt[0], 0);
    ASSERT_EQ(vecInt[1], 1);
    ASSERT_EQ(vecInt[2], 2);

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

    const std::array<unsigned long, 3> arrUnsignedLong = {0, 1, 2};
    vecUnsignedLong                                    = Vec3Dul(arrUnsignedLong);
    ASSERT_EQ(vecUnsignedLong[0], 0);
    ASSERT_EQ(vecUnsignedLong[1], 1);
    ASSERT_EQ(vecUnsignedLong[2], 2);
}

TEST(TestVector3d, testAssignmentOperator)
{
    Vec3D vec1(0.0, 1.0, 2.0);
    Vec3D vec2(1.0, 2.0, 3.0);

    vec1 = vec2;

    ASSERT_EQ(vec1, vec2);
}

TEST(TestVector3d, testAdditionOperator)
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
}

TEST(TestVector3d, testSubtractionOperator)
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
}

TEST(TestVector3d, testMultiplicationOperator)
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
}

TEST(TestVector3d, testDivisionOperator)
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
}

TEST(TestVector3d, testIsLessOperator)
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

TEST(TestVector3d, testIsGreaterOperator)
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

TEST(TestVector3d, testFABS)
{
    const auto vec1 = Vec3D(0.0, -1.0, 2.0);
    EXPECT_EQ(fabs(vec1), Vec3D(0.0, 1.0, 2.0));
}

TEST(TestVector3d, testStaticCast)
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

TEST(TestVector3d, testRound)
{
    const auto vec = Vec3D(0.0, 1.4, 2.8);

    EXPECT_EQ(round(vec), Vec3D(0.0, 1.0, 3.0));
}

TEST(TestVector3d, testCeil)
{
    const auto vec = Vec3D(0.0, 1.4, 2.8);

    EXPECT_EQ(ceil(vec), Vec3D(0.0, 2.0, 3.0));
}

TEST(TestVector3d, testFloor)
{
    const auto vec = Vec3D(0.0, 1.4, 2.8);

    EXPECT_EQ(floor(vec), Vec3D(0.0, 1.0, 2.0));
}

TEST(TestVector3d, testNorm)
{
    const auto vec = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(norm(vec), sqrt(5.0));
}

TEST(TestVector3d, testNormSquared)
{
    const auto vec = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(normSquared(vec), 5.0);
}

TEST(TestVector3d, testMinimum)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(minimum(vec1), 0.0);

    const auto vec2 = Vec3Di(0, 1, 2);

    EXPECT_EQ(minimum(vec2), 0);

    const auto vec3 = Vec3Dul(0, 1, 2);

    EXPECT_EQ(minimum(vec3), 0);
}

TEST(TestVector3d, testSum)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(sum(vec1), 3.0);

    const auto vec2 = Vec3Di(0, 1, 2);

    EXPECT_EQ(sum(vec2), 3);

    const auto vec3 = Vec3Dul(0, 1, 2);

    EXPECT_EQ(sum(vec3), 3);
}

TEST(TestVector3d, mean)
{
    const auto vec1 = Vec3D(0.0, 1.0, 2.0);

    EXPECT_EQ(mean(vec1), 1.0);
}

TEST(TestVector3d, testProduct)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(prod(vec1), 6.0);

    const auto vec2 = Vec3Di(1, 2, 3);

    EXPECT_EQ(prod(vec2), 6);

    const auto vec3 = Vec3Dul(1, 2, 3);

    EXPECT_EQ(prod(vec3), 6);
}

TEST(TestVector3d, scalarProduct)
{
    const auto vec1 = Vec3D(1.0, 2.0, 3.0);
    const auto vec2 = Vec3D(1.0, 2.0, 3.0);

    EXPECT_EQ(dot(vec1, vec2), 14.0);
}

TEST(TestVector3d, testOsStream)
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

TEST(TestVector3d, testBegin)
{
    const auto vec = Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(*vec.begin(), 1.0);

    const auto vec2 = Vec3Di(1, 2, 3);
    EXPECT_EQ(*vec2.begin(), 1);

    const auto vec3 = Vec3Dul(1, 2, 3);
    EXPECT_EQ(*vec3.begin(), 1);
}

TEST(TestVector3d, testEnd)
{
    const auto vec = Vec3D(1.0, 2.0, 3.0);
    EXPECT_EQ(*(vec.end() - 1), 3.0);

    const auto vec2 = Vec3Di(1, 2, 3);
    EXPECT_EQ(*(vec2.end() - 1), 3);

    const auto vec3 = Vec3Dul(1, 2, 3);
    EXPECT_EQ(*(vec3.end() - 1), 3);
}

TEST(TestVector3d, testEqualOperator)
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

TEST(TestVector3d, testAdditionAssignmentOperator)
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
}

TEST(TestVector3d, testSubtractionAssignmentOperator)
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
}

TEST(TestVector3d, testMultiplicationAssignmentOperator)
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
}

TEST(TestVector3d, testDivisionAssignmentOperator)
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
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}