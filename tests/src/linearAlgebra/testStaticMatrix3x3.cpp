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

#include <iosfwd>

#include "matrixNear.hpp"
#include "staticMatrix.hpp"

TEST(TestStaticMatrix3x3, unaryMinusOperator)
{
    linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };

    EXPECT_EQ(
        -mat,
        linalg::StaticMatrix3x3<double>(
            {-1.0, -2.0, -3.0},
            {-4.0, -5.0, -6.0},
            {-7.0, -8.0, -9.0}
        )
    );
}

TEST(TestStaticMatrix3x3, subtractMatrices)
{
    const linalg::StaticMatrix3x3<double> lhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };
    const linalg::StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };

    EXPECT_EQ(
        lhs - rhs,
        linalg::StaticMatrix3x3<double>(
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0}
        )
    );
}

TEST(TestStaticMatrix3x3, addAssignmentOperator)
{
    linalg::StaticMatrix3x3<double> lhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };
    const linalg::StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };

    lhs += rhs;

    EXPECT_EQ(
        lhs,
        linalg::StaticMatrix3x3<double>(
            {2.0, 4.0, 6.0},
            {8.0, 10.0, 12.0},
            {14.0, 16.0, 18.0}
        )
    );
}

TEST(TestStaticMatrix3x3, addMatrices)
{
    const linalg::StaticMatrix3x3<double> lhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };
    const linalg::StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7, 8, 9}
    };

    EXPECT_EQ(
        lhs + rhs,
        linalg::StaticMatrix3x3<double>(
            {2.0, 4.0, 6.0},
            {8.0, 10.0, 12.0},
            {14.0, 16.0, 18.0}
        )
    );
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrices)
{
    const linalg::StaticMatrix3x3<double> lhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    const linalg::StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    EXPECT_EQ(
        lhs * rhs,
        linalg::StaticMatrix3x3<double>(
            {30.0, 36.0, 42.0},
            {66.0, 81.0, 96.0},
            {102.0, 126.0, 150.0}
        )
    );
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrixWithScalar)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const double scalar = 3.0;

    EXPECT_EQ(
        mat * scalar,
        linalg::StaticMatrix3x3<double>(
            {3.0, 6.0, 9.0},
            {12.0, 15.0, 18.0},
            {21.0, 24.0, 27.0}
        )
    );
    EXPECT_EQ(scalar * mat, mat * scalar);
}

TEST(TestStaticMatrix3x3, addStaticMatrixWithScalar)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const double scalar = 3.0;

    EXPECT_EQ(
        mat + scalar,
        linalg::StaticMatrix3x3<double>(
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        )
    );
}

TEST(TestStaticMatrix3x3, multiplyStaticMatrixWithVector3D)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const linalg::Vec3D vec{1.0, 2.0, 3.0};

    EXPECT_EQ(mat * vec, linalg::Vec3D(14.0, 32.0, 50.0));
}

TEST(TestStaticMatrix3x3, transpose)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    EXPECT_EQ(
        transpose(mat),
        linalg::StaticMatrix3x3<double>(
            {1.0, 4.0, 7.0},
            {2.0, 5.0, 8.0},
            {3.0, 6.0, 9.0}
        )
    );
}

TEST(TestStaticMatrix3x3, determinant)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(det(mat), 45.0);
}

TEST(TestStaticMatrix3x3, vectorProductToStaticMatrix3x3)
{
    const linalg::Vec3D lhs{1.0, 2.0, 3.0};
    const linalg::Vec3D rhs{4.0, 5.0, 6.0};

    EXPECT_EQ(
        tensorProduct(lhs, rhs),
        linalg::StaticMatrix3x3<double>(
            {4.0, 5.0, 6.0},
            {8.0, 10.0, 12.0},
            {12.0, 15.0, 18.0}
        )
    );
}

TEST(TestStaticMatrix3x3, outputStreamOperator)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    std::stringstream sstream;
    sstream << mat;

    EXPECT_EQ(sstream.str(), "[[1 2 3]\n [4 5 6]\n [7 8 9]]");
}

TEST(TestStaticMatrix3x3, cofactorMatrix)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(
        cofactorMatrix(mat),
        linalg::StaticMatrix3x3<double>(
            {-17.0, -2.0, 22.0},
            {13.0, -17.0, 7.0},
            {-2.0, 13.0, -8.0}
        )
    );
}

TEST(TestStaticMatrix3x3, inverse)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_MATRIX_NEAR(
        inverse(mat),
        linalg::StaticMatrix3x3<double>(
            {-0.377777777777778, 0.288888888888889, -0.0444444444444444},
            {-0.0444444444444444, -0.377777777777778, 0.288888888888889},
            {0.488888888888889, 0.155555555555556, -0.177777777777778}
        ),
        1e-8
    );
}

TEST(TestStaticMatrix3x3, diagonalOfMatrix)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(diagonal(mat), linalg::Vec3D(1.0, 4.0, 7.0));
}

TEST(TestStaticMatrix3x3, diagonalMatrixOfVec3D)
{
    const linalg::Vec3D vec{1.0, 2.0, 3.0};

    EXPECT_MATRIX_NEAR(
        linalg::diagonalMatrix(vec),
        linalg::StaticMatrix3x3<double>(
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
        linalg::diagonalMatrix(1.0),
        linalg::StaticMatrix3x3<double>(
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        ),
        1e-50
    );
}

TEST(TestStaticMatrix3x3, trace)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(trace(mat), 12.0);
}

TEST(TestStaticMatrix3x3, multiplyAssignmentOperatorScalar)
{
    linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    mat *= 2.0;

    EXPECT_EQ(
        mat,
        linalg::StaticMatrix3x3<double>(
            {2.0, 4.0, 6.0},
            {12.0, 8.0, 10.0},
            {16.0, 18.0, 14.0}
        )
    );
}

TEST(TestStaticMatrix3x3, divideAssignmentOperatorScalar)
{
    linalg::StaticMatrix3x3<double> mat{
        {2.0, 4.0, 6.0},
        {12.0, 8.0, 10.0},
        {16.0, 18.0, 14.0}
    };

    mat /= 2.0;

    EXPECT_EQ(
        mat,
        linalg::StaticMatrix3x3<double>(
            {1.0, 2.0, 3.0},
            {6.0, 4.0, 5.0},
            {8.0, 9.0, 7.0}
        )
    );
}

TEST(TestStaticMatrix3x3, subtractionAssignmentOperatorMatrices)
{
    linalg::StaticMatrix3x3<double> lhs{
        {2.0, 4.0, 6.0},
        {12.0, 8.0, 10.0},
        {16.0, 18.0, 14.0}
    };
    const linalg::StaticMatrix3x3<double> rhs{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    lhs -= rhs;

    EXPECT_EQ(
        lhs,
        linalg::StaticMatrix3x3<double>(
            {1.0, 2.0, 3.0},
            {6.0, 4.0, 5.0},
            {8.0, 9.0, 7.0}
        )
    );
}

TEST(TestStaticMatrix3x3, getDiagonalVectorFromMatrix)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 2.0, 3.0},
        {6.0, 4.0, 5.0},
        {8.0, 9.0, 7.0}
    };

    EXPECT_EQ(diagonal(mat), linalg::Vec3D(1.0, 4.0, 7.0));
}

TEST(TestStaticMatrix3x3, getExponentialMatrix)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    };

    EXPECT_MATRIX_NEAR(
        exp(mat),
        linalg::StaticMatrix3x3<double>(
            {exp(1.0), exp(0.0), exp(0.0)},
            {exp(0.0), exp(2.0), exp(0.0)},
            {exp(0.0), exp(0.0), exp(3.0)}
        ),
        1e-6
    );
}

TEST(TestStaticMatrix3x3, getKroneckerDeltaMatrix)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    auto delta = linalg::kroneckerDeltaMatrix<double>();

    EXPECT_EQ(delta, mat);
}

TEST(TestStaticMatrix3x3, getExponentialPadeMatrix)
{
    const linalg::StaticMatrix3x3<double> mat{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    EXPECT_MATRIX_NEAR(
        expPade(mat),
        linalg::StaticMatrix3x3<double>(
            {exp(1.0), 0.0, 0.0},
            {0.0, exp(1.0), 0.0},
            {0.0, 0.0, exp(1.0)}
        ),
        1e-3
    );

    const linalg::StaticMatrix3x3<double> mat2{
        {1.0, 1.0, 1.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    EXPECT_MATRIX_NEAR(
        expPade(mat2),
        linalg::StaticMatrix3x3<double>(
            {exp(1.0), exp(1.0), exp(1.0)},
            {0.0, exp(1.0), 0.0},
            {0.0, 0.0, exp(1.0)}
        ),
        1e-3
    );
}
