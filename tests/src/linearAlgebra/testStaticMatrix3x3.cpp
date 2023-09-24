#include "matrixNear.hpp"             // for EXPECT_MATRIX_NEAR
#include "staticMatrix3x3.hpp"        // for diagonalMatrix, inverse, operator*
#include "staticMatrix3x3Class.hpp"   // for StaticMatrix3x3
#include "vector3d.hpp"               // for Vec3D, linearAlgebra

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), TEST
#include <iosfwd>          // for stringstream, ostream
#include <memory>          // for allocator

using namespace linearAlgebra;

TEST(TestStaticMatrix3x3, unaryMinusOperator)
{
    StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};

    EXPECT_EQ(-mat, StaticMatrix3x3<double>({-1.0, -2.0, -3.0}, {-4.0, -5.0, -6.0}, {-7.0, -8.0, -9.0}));
}

TEST(TestStaticMatrix3x3, addAssignmentOperator)
{
    StaticMatrix3x3<double>       lhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};
    const StaticMatrix3x3<double> rhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};

    lhs += rhs;

    EXPECT_EQ(lhs, StaticMatrix3x3<double>({2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}, {14.0, 16.0, 18.0}));
}

TEST(TestStaticMatrix3x3, addMatrices)
{
    const StaticMatrix3x3<double> lhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};
    const StaticMatrix3x3<double> rhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};

    EXPECT_EQ(lhs + rhs, StaticMatrix3x3<double>({2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}, {14.0, 16.0, 18.0}));
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

TEST(TestStaticMatrix3x3, multiplyStaticMatrixWithVector3D)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    const Vec3D vec{1.0, 2.0, 3.0};

    EXPECT_EQ(mat * vec, Vec3D(14.0, 32.0, 50.0));
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

TEST(TestStaticMatrix3x3, cofactorMatrix)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {6.0, 4.0, 5.0}, {8.0, 9.0, 7.0}};

    EXPECT_EQ(cofactorMatrix(mat), StaticMatrix3x3<double>({-17.0, -2.0, 22.0}, {13.0, -17.0, 7.0}, {-2.0, 13.0, -8.0}));
}

TEST(TestStaticMatrix3x3, inverse)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {6.0, 4.0, 5.0}, {8.0, 9.0, 7.0}};

    EXPECT_MATRIX_NEAR(inverse(mat),
                       StaticMatrix3x3<double>({-0.377777777777778, 0.288888888888889, -0.0444444444444444},
                                               {-0.0444444444444444, -0.377777777777778, 0.288888888888889},
                                               {0.488888888888889, 0.155555555555556, -0.177777777777778}),
                       1e-8);
}

TEST(TestStaticMatrix3x3, diagonalMatrixofVec3D)
{
    const Vec3D vec{1.0, 2.0, 3.0};

    EXPECT_MATRIX_NEAR(diagonalMatrix(vec), StaticMatrix3x3<double>({1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}), 1e-50);
}

TEST(TestStaticMatrix3x3, diagonalMatrixOfScalar)
{
    EXPECT_MATRIX_NEAR(diagonalMatrix(1.0), StaticMatrix3x3<double>({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}), 1e-50);
}

TEST(TestStaticMatrix3x3, trace)
{
    const StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {6.0, 4.0, 5.0}, {8.0, 9.0, 7.0}};

    EXPECT_EQ(trace(mat), 12.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}