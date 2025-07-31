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

#ifndef _EXPECT_VECTOR_NEAR_HPP_

#define _EXPECT_VECTOR_NEAR_HPP_

#define EXPECT_VECTOR_NEAR(vector1, vector2, value) \
    EXPECT_NEAR(vector1[0], vector2[0], value);     \
    EXPECT_NEAR(vector1[1], vector2[1], value);     \
    EXPECT_NEAR(vector1[2], vector2[2], value);

#define ASSERT_VECTOR_NEAR(vector1, vector2) \
    EXPECT_VECTOR_NEAR(vector1, vector2)

#endif   // _EXPECT_VECTOR_NEAR_HPP_