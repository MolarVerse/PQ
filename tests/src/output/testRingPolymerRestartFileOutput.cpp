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

#include "testRingPolymerRestartFileOutput.hpp"

#include "ringPolymerSettings.hpp"   // for RingPolymerSettings

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, InitGoogleTest, RUN_ALL_TESTS
#include <iosfwd>          // for ifstream
#include <string>          // for getline, string

/**
 * @brief tests writing restart file for ring polymer
 *
 */
TEST_F(TestRingPolymerRestartFileOutput, write)
{
    settings::RingPolymerSettings::setNumberOfBeads(_beads.size());

    _rstFileOutput->setFilename("default.rpmd.rst");
    _rstFileOutput->write(_beads, 10);
    _rstFileOutput->close();
    std::ifstream file("default.rpmd.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line,
              "    H1\t    1\t    1\t     1.00000000\t     1.00000000\t     1.00000000\t     1.00000000e+00\t     "
              "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "    O1\t    2\t    1\t     1.00000000\t     2.00000000\t     3.00000000\t     3.00000000e+00\t     "
              "4.00000000e+00\t     5.00000000e+00\t     2.00000000\t     3.00000000\t     4.00000000");

    getline(file, line);
    EXPECT_EQ(line,
              "   Ar1\t    1\t    2\t     1.00000000\t     1.00000000\t     1.00000000\t     1.00000000e+00\t     "
              "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "    H2\t    1\t    1\t     2.00000000\t     2.00000000\t     2.00000000\t     2.00000000e+00\t     "
              "2.00000000e+00\t     2.00000000e+00\t     2.00000000\t     2.00000000\t     2.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "    O2\t    2\t    1\t     2.00000000\t     3.00000000\t     4.00000000\t     4.00000000e+00\t     "
              "5.00000000e+00\t     6.00000000e+00\t     3.00000000\t     4.00000000\t     5.00000000");

    getline(file, line);
    EXPECT_EQ(line,
              "   Ar2\t    1\t    2\t     2.00000000\t     2.00000000\t     2.00000000\t     2.00000000e+00\t     "
              "2.00000000e+00\t     2.00000000e+00\t     2.00000000\t     2.00000000\t     2.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}