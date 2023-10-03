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

#include "testRstFileOutput.hpp"

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing the restart file
 *
 */
TEST_F(TestRstFileOutput, write)
{
    _rstFileOutput->setFilename("default.rst");
    _rstFileOutput->write(*_simulationBox, 10);
    _rstFileOutput->close();
    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10");
    getline(file, line);
    EXPECT_EQ(line, "Box   10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line,
              "H    \t1    \t1    \t     1.00000000\t     1.00000000\t     1.00000000\t     1.00000000e+00\t     "
              "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line,
              "O    \t2    \t1    \t     1.00000000\t     2.00000000\t     3.00000000\t     3.00000000e+00\t     "
              "4.00000000e+00\t     5.00000000e+00\t     2.00000000\t     3.00000000\t     4.00000000");

    getline(file, line);
    EXPECT_EQ(line,
              "Ar   \t1    \t2    \t     1.00000000\t     1.00000000\t     1.00000000\t     1.00000000e+00\t     "
              "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     1.00000000\t     1.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}