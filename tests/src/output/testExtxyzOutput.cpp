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

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "extxyzOutput.hpp"
#include "frame.hpp"
#include "testTrajectoryOutput.hpp"
#include "units.hpp"

/**
 * @brief one frame in PQ units with cell, step and energy in the comment
 *
 */
TEST_F(TestTrajectoryOutput, writeExtxyz)
{
    out::ExtxyzOutput output{"default.xyz"};
    output.setFilename("default.xyz", false);

    auto frame            = out::Frame{*_simulationBox};
    frame.potentialEnergy = -1.5 * kcal_per_mol;

    output.write(frame, 7);
    output.close();

    std::ifstream file("default.xyz");
    std::string   line;

    getline(file, line);
    EXPECT_EQ(line, "3");
    getline(file, line);
    EXPECT_EQ(
        line,
        R"(Lattice="10.00000000 0.00000000 0.00000000 0.00000000 10.00000000 0.00000000 0.00000000 0.00000000 10.00000000" Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3:charge:R:1 pbc="T T T" step=7 energy=-1.5000000000 units=pq)"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "H          1.00000000      1.00000000      1.00000000       "
        "1.00000000e+00       1.00000000e+00       1.00000000e+00      "
        "1.00000000      1.00000000      1.00000000      1.00000000"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "O          1.00000000      2.00000000      3.00000000       "
        "3.00000000e+00       4.00000000e+00       5.00000000e+00      "
        "2.00000000      3.00000000      4.00000000     -1.00000000"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "Ar         1.00000000      1.00000000      1.00000000       "
        "1.00000000e+00       1.00000000e+00       1.00000000e+00      "
        "1.00000000      1.00000000      1.00000000      0.00000000"
    );
    EXPECT_FALSE(getline(file, line));
}
