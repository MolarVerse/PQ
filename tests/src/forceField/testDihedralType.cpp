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

#include <gtest/gtest.h>   // for Test, EXPECT_FALSE, InitGoogleTest, RUN_...

#include <string>   // for allocator, string

#include "dihedralType.hpp"   // for DihedralType
#include "gtest/gtest.h"      // for AssertionResult, Message, TestPartResult

/**
 * @brief tests operator== for DihedralType
 *
 */
TEST(TestDihedralType, operatorEqual)
{
    forceField::DihedralType dihedralType1(0, 1.0, 2.0, 3.0);
    forceField::DihedralType dihedralType2(0, 1.0, 2.0, 3.0);
    forceField::DihedralType dihedralType3(1, 1.0, 2.0, 3.0);
    forceField::DihedralType dihedralType4(0, 2.0, 2.0, 3.0);
    forceField::DihedralType dihedralType5(0, 1.0, 3.0, 3.0);
    forceField::DihedralType dihedralType6(0, 1.0, 2.0, 4.0);

    EXPECT_TRUE(dihedralType1 == dihedralType2);
    EXPECT_FALSE(dihedralType1 == dihedralType3);
    EXPECT_FALSE(dihedralType1 == dihedralType4);
    EXPECT_FALSE(dihedralType1 == dihedralType5);
    EXPECT_FALSE(dihedralType1 == dihedralType6);
}