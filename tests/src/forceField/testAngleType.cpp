/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "angleType.hpp"   // for AngleType

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include <gtest/gtest.h>   // for Test, EXPECT_FALSE, InitGoogleTest, RUN_ALL...
#include <string>          // for allocator, string

/**
 * @brief tests operator== for AngleType
 *
 */
TEST(TestAngleType, operatorEqual)
{
    forceField::AngleType angleType1(0, 1.0, 2.0);
    forceField::AngleType angleType2(0, 1.0, 2.0);
    forceField::AngleType angleType3(1, 1.0, 2.0);
    forceField::AngleType angleType4(0, 2.0, 2.0);
    forceField::AngleType angleType5(0, 1.0, 3.0);

    EXPECT_TRUE(angleType1 == angleType2);
    EXPECT_FALSE(angleType1 == angleType3);
    EXPECT_FALSE(angleType1 == angleType4);
    EXPECT_FALSE(angleType1 == angleType5);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}