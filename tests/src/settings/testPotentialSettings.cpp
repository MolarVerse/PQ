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

#include "potentialSettings.hpp"   // for string, PotentialSettings, NonCoulo

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, Test, TestInfo (ptr only)
#include <memory>          // for allocator

/**
 * @brief tests string function of enum nonCoulombType
 *
 */
TEST(TestPotentialSettings, string_nonCoulombType)
{
    EXPECT_EQ(settings::string(settings::NonCoulombType::LJ), "lj");
    EXPECT_EQ(settings::string(settings::NonCoulombType::LJ_9_12), "lj_9_12");
    EXPECT_EQ(settings::string(settings::NonCoulombType::BUCKINGHAM), "buck");
    EXPECT_EQ(settings::string(settings::NonCoulombType::MORSE), "morse");
    EXPECT_EQ(settings::string(settings::NonCoulombType::GUFF), "guff");
    EXPECT_EQ(settings::string(settings::NonCoulombType::NONE), "none");
}

/**
 * @brief tests setNonCoulombType function
 *
 */
TEST(TestPotentialSettings, setNonCoulombType)
{
    settings::PotentialSettings::setNonCoulombType("lj");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::LJ);

    settings::PotentialSettings::setNonCoulombType("lj_9_12");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::LJ_9_12);

    settings::PotentialSettings::setNonCoulombType("buck");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::BUCKINGHAM);

    settings::PotentialSettings::setNonCoulombType("morse");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::MORSE);

    settings::PotentialSettings::setNonCoulombType("guff");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::GUFF);

    settings::PotentialSettings::setNonCoulombType("none");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::NONE);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}