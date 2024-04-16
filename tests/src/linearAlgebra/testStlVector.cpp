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

#include "stlVector.hpp"   // for max, mean, sum

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <algorithm>       // for copy
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), EXPECT_EQ, InitGoogleTest, RUN_ALL_TESTS
#include <vector>          // for allocator, vector

TEST(TestStlVector, sum)
{
    const std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(sum(v1), 10.0);
}

TEST(TestStlVector, mean)
{
    const std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(mean(v1), 2.5);
}

TEST(TestStlVector, max)
{
    const std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(max(v1), 4.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}