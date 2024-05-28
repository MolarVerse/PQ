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

#include <gtest/gtest.h>   // for Test, EXPECT_FALSE, InitGoogleTest, RUN_ALL...

#include <string>   // for allocator, string

#include "bondType.hpp"    // for BondType
#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult

/**
 * @brief tests operator== for BondType
 *
 */
TEST(TestBondType, operatorEqual)
{
    forceField::BondType bondType1(0, 1.0, 2.0);
    forceField::BondType bondType2(0, 1.0, 2.0);
    forceField::BondType bondType3(1, 1.0, 2.0);
    forceField::BondType bondType4(0, 2.0, 2.0);
    forceField::BondType bondType5(0, 1.0, 3.0);

    EXPECT_TRUE(bondType1 == bondType2);
    EXPECT_FALSE(bondType1 == bondType3);
    EXPECT_FALSE(bondType1 == bondType4);
    EXPECT_FALSE(bondType1 == bondType5);
}