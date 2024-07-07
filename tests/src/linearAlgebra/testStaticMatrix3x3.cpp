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

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), TEST

#include <iosfwd>   // for stringstream, ostream
#include <memory>   // for allocator

#include "gtest/gtest.h"      // for Message, TestPartResult
#include "matrixNear.hpp"     // for EXPECT_MATRIX_NEAR
#include "staticMatrix.hpp"   // for diagonalMatrix, inverse, operator*
#include "vector3d.hpp"       // for Vec3D, linearAlgebra

using namespace linearAlgebra;

TEST(TestStaticMatrix3x3, unaryMinusOperator)
{
    StaticMatrix3x3<double> mat{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};

    EXPECT_EQ(
        -mat,
        StaticMatrix3x3<double>(
            {-1.0, -2.0, -3.0},
            {-4.0, -5.0, -6.0},
            {-7.0, -8.0, -9.0}
        )
    );
}

TEST(TestStaticMatrix3x3, addAssignmentOperator)
{
    StaticMatrix3x3<double> lhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7, 8, 9}};
    const StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };

    lhs += rhs;

    EXPECT_EQ(
        lhs,
        StaticMatrix3x3<double>(
            {2.0, 4.0, 6.0},
            {8.0, 10.0, 12.0},
            {14.0, 16.0, 18.0}
        )
    );
}

TEST(TestStaticMatrix3x3, addMatrices)
{
    const StaticMatrix3x3<double> lhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };
    const StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };

    EXPECT_EQ(
        lhs + rhs,
        StaticMatrix3x3<double>(
            {2.0, 4.0, 6.0},
            {8.0, 10.0, 12.0},
            {14.0, 16.0, 18.0}
        )
    );
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrices)
{
    const StaticMatrix3x3<double> lhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    const StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    EXPECT_EQ(
        lhs * rhs,
        StaticMatrix3x3<double>(
            {30.0, 36.0, 42.0},
            {66.0, 81.0, 96.0},
            {102.0, 126.0, 150.0}
        )
    );
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrixWithScalar)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const double scalar = 3.0;

    EXPECT_EQ(
        mat * scalar,
        StaticMatrix3x3<double>(
            {3.0, 6.0, 9.0},
            {12.0, 15.0, 18.0},
            {21.0, 24.0, 27.0}
        )
    );
    EXPECT_EQ(scalar * mat, mat * scalar);
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrixWithVector3D)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const Vec3D vec{1.0, 2.0, 3.0};

    EXPECT_EQ(mat * vec, Vec3D(14.0, 32.0, 50.0));
}

TEST(TestStaticMatrix3x3, transpose)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    EXPECT_EQ(
        transpose(mat),
        StaticMatrix3x3<double>(
            {1.0, 4.0, 7.0},
            {2.0, 5.0, 8.0},
            {3.0, 6.0, 9.0}
        )
    );
}

TEST(TestStaticMatrix3x3, determinant)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(det(mat), 45.0);
}

TEST(TestStaticMatrix3x3, vectorProductToStaticMatrix3x3)
{
    const Vec3D lhs{1.0, 2.0, 3.0};
    const Vec3D rhs{4.0, 5.0, 6.0};

    EXPECT_EQ(
        tensorProduct(lhs, rhs),
        StaticMatrix3x3<double>(
            {4.0, 5.0, 6.0},
            {8.0, 10.0, 12.0},
            {12.0, 15.0, 18.0}
        )
    );
}

TEST(TestStaticMatrix3x3, outputStreamOperator)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    std::stringstream ss;
    ss << mat;

    EXPECT_EQ(ss.str(), "[[1 2 3]\n [4 5 6]\n [7 8 9]]");
}

TEST(TestStaticMatrix3x3, cofactorMatrix)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(
        cofactorMatrix(mat),
        StaticMatrix3x3<double>(
            {-17.0, -2.0, 22.0},
            {13.0, -17.0, 7.0},
            {-2.0, 13.0, -8.0}
        )
    );
}

TEST(TestStaticMatrix3x3, inverse)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_MATRIX_NEAR(
        inverse(mat),
        StaticMatrix3x3<double>(
            {-0.377777777777778, 0.288888888888889, -0.0444444444444444},
            {-0.0444444444444444, -0.377777777777778, 0.288888888888889},
            {0.488888888888889, 0.155555555555556, -0.177777777777778}
        ),
        1e-8
    );
}

TEST(TestStaticMatrix3x3, diagonalOfMatrix)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(diagonal(mat), Vec3D(1.0, 4.0, 7.0));
}

TEST(TestStaticMatrix3x3, diagonalMatrixOfVec3D)
{
    const Vec3D vec{1.0, 2.0, 3.0};

    EXPECT_MATRIX_NEAR(
        diagonalMatrix(vec),
        StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 2.0, 0.0},
            {0.0, 0.0, 3.0}
        ),
        1e-50
    );
}

TEST(TestStaticMatrix3x3, diagonalMatrixOfScalar)
{
    EXPECT_MATRIX_NEAR(
        diagonalMatrix(1.0),
        StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        ),
        1e-50
    );
}

TEST(TestStaticMatrix3x3, trace)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(trace(mat), 12.0);
}

TEST(TestStaticMatrix3x3, multiplyAssignmentOperator_scalar)
{
    StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    mat *= 2.0;

    EXPECT_EQ(
        mat,
        StaticMatrix3x3<double>(
            {2.0, 4.0, 6.0},
            {12.0, 8.0, 10.0},
            {16.0, 18.0, 14.0}
        )
    );
}

TEST(TestStaticMatrix3x3, divideAssignmentOperator_scalar)
{
    StaticMatrix3x3<double> mat{
        {2.0, 4.0, 6.0},
        {12.0, 8.0, 10.0},
        {16.0, 18.0, 14.0}
    };

    mat /= 2.0;

    EXPECT_EQ(
        mat,
        StaticMatrix3x3<double>(
            {1.0, 2.0, 3.0},
            {6.0, 4.0, 5.0},
            {8.0, 9.0, 7.0}
        )
    );
}

TEST(TestStaticMatrix3x3, subtractionAssignmentOperator_matrices)
{
    StaticMatrix3x3<double> lhs{
        {2.0, 4.0, 6.0},
        {12.0, 8.0, 10.0},
        {16.0, 18.0, 14.0}
    };
    const StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    lhs -= rhs;

    EXPECT_EQ(
        lhs,
        StaticMatrix3x3<double>(
            {1.0, 2.0, 3.0},
            {6.0, 4.0, 5.0},
            {8.0, 9.0, 7.0}
        )
    );
}

TEST(TestStaticMatrix3x3, getDiagonalVectorFromMatrix)
{
    const StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(diagonal(mat), Vec3D(1.0, 4.0, 7.0));
}