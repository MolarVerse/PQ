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

#include "testTrajectoryOutput.hpp"

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief Test the writeXyz method
 *
 */
TEST_F(TestTrajectoryOutput, writeXyz)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeXyz(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "H    \t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O    \t     1.00000000\t     2.00000000\t     3.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t     1.00000000\t     1.00000000\t     1.00000000");
}

/**
 * @brief Test the writeVelocities method
 *
 */
TEST_F(TestTrajectoryOutput, writeVelocities)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeVelocities(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "H    \t      1.00000000e+00\t      1.00000000e+00\t      1.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "O    \t      3.00000000e+00\t      4.00000000e+00\t      5.00000000e+00");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t      1.00000000e+00\t      1.00000000e+00\t      1.00000000e+00");
}

/**
 * @brief Test the writeForces method
 *
 */
TEST_F(TestTrajectoryOutput, writeForces)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeForces(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "# Total force = 8.77496e+00 kcal/mol/Angstrom");
    getline(file, line);
    EXPECT_EQ(line, "H    \t     1.00000000\t     1.00000000\t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O    \t     2.00000000\t     3.00000000\t     4.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t     1.00000000\t     1.00000000\t     1.00000000");
}

/**
 * @brief Test the writeXyz method
 *
 */
TEST_F(TestTrajectoryOutput, writeCharges)
{
    _trajectoryOutput->setFilename("default.xyz");
    _trajectoryOutput->writeCharges(*_simulationBox);
    _trajectoryOutput->close();
    std::ifstream file("default.xyz");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "3  10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "H    \t     1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "O    \t    -1.00000000");
    getline(file, line);
    EXPECT_EQ(line, "Ar   \t     0.00000000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}