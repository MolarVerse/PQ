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

#ifndef _EXPECT_MATRIX_NEAR_HPP_

#define _EXPECT_MATRIX_NEAR_HPP_

#define EXPECT_MATRIX_NEAR(matrix1, matrix2, value)                                                                              \
    EXPECT_NEAR(matrix1[0][0], matrix2[0][0], value);                                                                            \
    EXPECT_NEAR(matrix1[0][1], matrix2[0][1], value);                                                                            \
    EXPECT_NEAR(matrix1[0][2], matrix2[0][2], value);                                                                            \
    EXPECT_NEAR(matrix1[1][0], matrix2[1][0], value);                                                                            \
    EXPECT_NEAR(matrix1[1][1], matrix2[1][1], value);                                                                            \
    EXPECT_NEAR(matrix1[1][2], matrix2[1][2], value);                                                                            \
    EXPECT_NEAR(matrix1[2][0], matrix2[2][0], value);                                                                            \
    EXPECT_NEAR(matrix1[2][1], matrix2[2][1], value);                                                                            \
    EXPECT_NEAR(matrix1[2][2], matrix2[2][2], value);

#define ASSERT_MATRIX_NEAR(matrix1, matrix2) EXPECT_MATRIX_NEAR(matrix1, matrix2)

#endif   // _EXPECT_MATRIX_NEAR_HPP_