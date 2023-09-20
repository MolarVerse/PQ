#include "staticMatrix3x3.hpp"
#include "vector3d.hpp"

#include <gtest/gtest.h>

using namespace linearAlgebra;

TEST(TestStaticMatrix3x3, addAssignmentOperator)
{
    StaticMatrix3x3<double>       lhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};
    const StaticMatrix3x3<double> rhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};

    lhs += rhs;

    EXPECT_EQ(lhs, StaticMatrix3x3<double>({2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}, {14.0, 16.0, 18.0}));
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrices)
{
    const StaticMatrix3x3<double> lhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    const StaticMatrix3x3<double> rhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    EXPECT_EQ(lhs * rhs, StaticMatrix3x3<double>({30.0, 36.0, 42.0}, {66.0, 81.0, 96.0}, {102.0, 126.0, 150.0}));
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrixWithScalar)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    const double scalar = 3.0;

    EXPECT_EQ(mat * scalar, StaticMatrix3x3<double>({3.0, 6.0, 9.0}, {12.0, 15.0, 18.0}, {21.0, 24.0, 27.0}));
    EXPECT_EQ(scalar * mat, mat * scalar);
}

TEST(TestStaticMatrix3x3, transpose)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    EXPECT_EQ(transpose(mat), StaticMatrix3x3<double>({1.0, 4.0, 7.0}, {2.0, 5.0, 8.0}, {3.0, 6.0, 9.0}));
}

TEST(TestStaticMatrix3x3, determinant)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {6.0, 4.0, 5.0}, {8.0, 9.0, 7.0}};

    EXPECT_EQ(det(mat), 45.0);
}

TEST(TestStaticMatrix3x3, vectorProductToStaticMatrix3x3)
{
    const Vec3D lhs{1.0, 2.0, 3.0};
    const Vec3D rhs{4.0, 5.0, 6.0};

    EXPECT_EQ(vectorProduct(lhs, rhs), StaticMatrix3x3<double>({4.0, 5.0, 6.0}, {8.0, 10.0, 12.0}, {12.0, 15.0, 18.0}));
}

TEST(TestStaticMatrix3x3, outputStreamOperator)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    std::stringstream ss;
    ss << mat;

    EXPECT_EQ(ss.str(), "[[1 2 3]\n [4 5 6]\n [7 8 9]]");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}