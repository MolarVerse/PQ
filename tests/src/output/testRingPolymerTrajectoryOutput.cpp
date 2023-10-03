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

#include "testRingPolymerTrajectoryOutput.hpp"

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief Test the writeXyz method
 *
 */
TEST_F(TestRingPolymerTrajectoryOutput, writeXyz)
{
    _trajectoryOutput->setFilename("default.rpmd.xyz");
    _trajectoryOutput->writeXyz(_beads);
    _trajectoryOutput->close();
    std::ifstream file("default.rpmd.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "6  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "    H1\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    O1\t     1.00000000\t     2.00000000\t     3.00000000");
    getline(file, line);
    EXPECT_EQ(line, "   Ar1\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    H2\t     2.00000000\t     2.00000000\t     2.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    O2\t     2.00000000\t     3.00000000\t     4.00000000");
    getline(file, line);
    EXPECT_EQ(line, "   Ar2\t     2.00000000\t     2.00000000\t     2.00000000");
}

/**
 * @brief Test the writeVelocities method
 *
 */
TEST_F(TestRingPolymerTrajectoryOutput, writeVelocities)
{
    _trajectoryOutput->setFilename("default.rpmd.xyz");
    _trajectoryOutput->writeVelocities(_beads);
    _trajectoryOutput->close();
    std::ifstream file("default.rpmd.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "6  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "    H1\t      1.00000000e+00\t      1.00000000e+00\t      1.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "    O1\t      3.00000000e+00\t      4.00000000e+00\t      5.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "   Ar1\t      1.00000000e+00\t      1.00000000e+00\t      1.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "    H2\t      2.00000000e+00\t      2.00000000e+00\t      2.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "    O2\t      4.00000000e+00\t      5.00000000e+00\t      6.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "   Ar2\t      2.00000000e+00\t      2.00000000e+00\t      2.00000000e+00");
}

/**
 * @brief Test the writeForces method
 *
 */
TEST_F(TestRingPolymerTrajectoryOutput, writeForces)
{
    _trajectoryOutput->setFilename("default.rpmd.xyz");
    _trajectoryOutput->writeForces(_beads);
    _trajectoryOutput->close();
    std::ifstream file("default.rpmd.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "6  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "# Total force = 2.27034e+01 kcal/mol/Angstrom");
    getline(file, line);
    EXPECT_EQ(line, "    H1\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    O1\t     2.00000000\t     3.00000000\t     4.00000000");
    getline(file, line);
    EXPECT_EQ(line, "   Ar1\t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    H2\t     2.00000000\t     2.00000000\t     2.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    O2\t     3.00000000\t     4.00000000\t     5.00000000");
    getline(file, line);
    EXPECT_EQ(line, "   Ar2\t     2.00000000\t     2.00000000\t     2.00000000");
}

/**
 * @brief Test the writeXyz method
 *
 */
TEST_F(TestRingPolymerTrajectoryOutput, writeCharges)
{
    _trajectoryOutput->setFilename("default.rpmd.xyz");
    _trajectoryOutput->writeCharges(_beads);
    _trajectoryOutput->close();
    std::ifstream file("default.rpmd.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "6  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "    H1\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    O1\t    -1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "   Ar1\t     0.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    H2\t     2.00000000");
    getline(file, line);
    EXPECT_EQ(line, "    O2\t     0.00000000");
    getline(file, line);
    EXPECT_EQ(line, "   Ar2\t     1.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}