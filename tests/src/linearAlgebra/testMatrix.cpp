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

#include <gtest/gtest.h>   // for Test, EXPECT_EQ, TestInfo (ptr only), InitG...

#include <memory>   // for allocator

#include "gtest/gtest.h"   // for Message, TestPartResult
#include "matrix.hpp"      // for Matrix, linearAlgebra

using namespace linearAlgebra;

/**
 * @brief tests constructors for Matrix
 *
 */
TEST(TestMatrix, constructors)
{
    Matrix<int> mat(2);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);

    Matrix<int> mat2(2, 3);
    EXPECT_EQ(mat2.rows(), 2);
    EXPECT_EQ(mat2.cols(), 3);
}

/**
 * @brief tests shape function for Matrix
 *
 */
TEST(TestMatrix, shape)
{
    auto mat = Matrix<int>(2, 3);

    const auto [rows, cols] = mat.shape();
    EXPECT_EQ(rows, 2);
    EXPECT_EQ(cols, 3);
}

/**
 * @brief tests size function for Matrix
 *
 */
TEST(TestMatrix, size)
{
    auto mat = Matrix<int>(2, 3);

    EXPECT_EQ(mat.size(), 6);
}