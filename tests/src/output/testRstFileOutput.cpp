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

#include "testRstFileOutput.hpp"

#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <string>

#include "exceptions.hpp"
#include "frame.hpp"
#include "noseHooverThermostat.hpp"
#include "thermostat.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "unit_types.hpp"
#include "units.hpp"

/**
 * @brief tests writing the restart file
 *
 */
TEST_F(TestRstFileOutput, write)
{
    _rstFileOutput->setFilename("default.rst", false);
    _rstFileOutput->write(
        out::Frame{*_simulationBox, thermostat::Thermostat()},
        10
    );
    _rstFileOutput->close();
    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10");
    getline(file, line);
    EXPECT_EQ(line, "Box   10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(
        line,
        "H    \t1    \t1    \t     1.00000000\t     1.00000000\t     "
        "1.00000000\t     1.00000000e+00\t     "
        "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     "
        "1.00000000\t     1.00000000"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "O    \t2    \t1    \t     1.00000000\t     2.00000000\t     "
        "3.00000000\t     3.00000000e+00\t     "
        "4.00000000e+00\t     5.00000000e+00\t     2.00000000\t     "
        "3.00000000\t     4.00000000"
    );

    getline(file, line);
    EXPECT_EQ(
        line,
        "Ar   \t1    \t2    \t     1.00000000\t     1.00000000\t     "
        "1.00000000\t     1.00000000e+00\t     "
        "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     "
        "1.00000000\t     1.00000000"
    );
}

/**
 * @brief tests writing the restart file
 *
 */
TEST_F(TestRstFileOutput, writeWithNoseHoover)
{
    NoseHooverSettings noseHooverSettings;
    noseHooverSettings.setNoseHooverCouplingFrequencyByUser(1.0e13 * Hz);
    const auto chis =
        std::map<size_t, double>{{0, 1.0}, {1, 2.0}, {2, 3.0}, {3, 0.0}};
    const auto zetas =
        std::map<size_t, double>{{0, 3.0}, {1, 2.0}, {2, 1.0}, {3, 0.0}};
    noseHooverSettings.setChiByUser(chis);
    noseHooverSettings.setZetaByUser(zetas);
    noseHooverSettings.setNoseHooverChainLengthByUser(3);
    auto thermostat = thermostat::NoseHooverThermostat(noseHooverSettings);

    _rstFileOutput->setFilename("default.rst", false);
    _rstFileOutput->write(out::Frame{*_simulationBox, thermostat}, 10);
    _rstFileOutput->close();
    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10");
    getline(file, line);
    EXPECT_EQ(line, "Box   10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "chi  1\t1.00000e+00\t3.00000e+00");
    getline(file, line);
    EXPECT_EQ(line, "chi  2\t2.00000e+00\t2.00000e+00");
    getline(file, line);
    EXPECT_EQ(line, "chi  3\t3.00000e+00\t1.00000e+00");
    getline(file, line);
    EXPECT_EQ(
        line,
        "H    \t1    \t1    \t     1.00000000\t     1.00000000\t     "
        "1.00000000\t     1.00000000e+00\t     "
        "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     "
        "1.00000000\t     1.00000000"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "O    \t2    \t1    \t     1.00000000\t     2.00000000\t     "
        "3.00000000\t     3.00000000e+00\t     "
        "4.00000000e+00\t     5.00000000e+00\t     2.00000000\t     "
        "3.00000000\t     4.00000000"
    );

    getline(file, line);
    EXPECT_EQ(
        line,
        "Ar   \t1    \t2    \t     1.00000000\t     1.00000000\t     "
        "1.00000000\t     1.00000000e+00\t     "
        "1.00000000e+00\t     1.00000000e+00\t     1.00000000\t     "
        "1.00000000\t     1.00000000"
    );
}

/**
 * @brief a failed write keeps the previous restart file and leaves no
 * temporary file behind on success
 *
 */
TEST_F(TestRstFileOutput, failedWriteKeepsPreviousFile)
{
    const out::Frame frame{*_simulationBox, thermostat::Thermostat()};

    _rstFileOutput->setFilename("default.rst", false);
    _rstFileOutput->write(frame, 10);
    EXPECT_FALSE(std::filesystem::exists("default.rst.tmp"));

    // a directory in the way makes writing the temporary file fail
    std::filesystem::create_directory("default.rst.tmp");

    EXPECT_THROW_MSG(
        _rstFileOutput->write(frame, 11),
        exc::InputFileException,
        "Could not write file - filename = default.rst.tmp"
    );

    std::filesystem::remove("default.rst.tmp");

    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10");
}
