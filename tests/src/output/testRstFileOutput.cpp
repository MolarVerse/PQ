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

#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
        10,
        100 * fs
    );
    _rstFileOutput->close();
    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10 Simulation_Time 0.1");
    getline(file, line);
    EXPECT_EQ(line, "Box   10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(
        line,
        "H    \t1    \t1    \t              1\t              1\t"
        "              1\t              1e+00\t              1e+00\t"
        "              1e+00\t              1\t              1\t"
        "              1"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "O    \t2    \t1    \t              1\t              2\t"
        "              3\t              3e+00\t              4e+00\t"
        "              5e+00\t              2\t              3\t"
        "              4"
    );

    getline(file, line);
    EXPECT_EQ(
        line,
        "Ar   \t1    \t2    \t              1\t              1\t"
        "              1\t              1e+00\t              1e+00\t"
        "              1e+00\t              1\t              1\t"
        "              1"
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
    _rstFileOutput
        ->write(out::Frame{*_simulationBox, thermostat}, 10, 100 * fs);
    _rstFileOutput->close();
    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10 Simulation_Time 0.1");
    getline(file, line);
    EXPECT_EQ(line, "Box   10 10 10  90 90 90");
    getline(file, line);
    EXPECT_EQ(line, "chi  1\t1\t3");
    getline(file, line);
    EXPECT_EQ(line, "chi  2\t2\t2");
    getline(file, line);
    EXPECT_EQ(line, "chi  3\t3\t1");
    getline(file, line);
    EXPECT_EQ(
        line,
        "H    \t1    \t1    \t              1\t              1\t"
        "              1\t              1e+00\t              1e+00\t"
        "              1e+00\t              1\t              1\t"
        "              1"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "O    \t2    \t1    \t              1\t              2\t"
        "              3\t              3e+00\t              4e+00\t"
        "              5e+00\t              2\t              3\t"
        "              4"
    );

    getline(file, line);
    EXPECT_EQ(
        line,
        "Ar   \t1    \t2    \t              1\t              1\t"
        "              1\t              1e+00\t              1e+00\t"
        "              1e+00\t              1\t              1\t"
        "              1"
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
    _rstFileOutput->write(frame, 10, 100 * fs);
    EXPECT_FALSE(std::filesystem::exists("default.rst.tmp"));

    // a directory in the way makes writing the temporary file fail
    std::filesystem::create_directory("default.rst.tmp");

    EXPECT_THROW_MSG(
        _rstFileOutput->write(frame, 11, 100 * fs),
        exc::InputFileException,
        "Could not write file - filename = default.rst.tmp"
    );

    std::filesystem::remove("default.rst.tmp");

    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "Step 10 Simulation_Time 0.1");
}

/**
 * @brief the box of a restart file reads back as the same numbers: a box that
 * is written with fewer digits changes the box of the restarted simulation
 *
 * @details the box of an NPT run, or a box taken from a start file with many
 * digits, is not a round number; a text with 6 digits (27.4421) differs from
 * the box of the run by about 4e-5 Angstrom
 */
TEST_F(TestRstFileOutput, boxIsWrittenWithoutLosingDigits)
{
    const auto length = 27.44205774028026567635;

    molsys::OrthorhombicBox orthoBox{{length, 0.5 * length, 1.0 / 3.0}};
    _simulationBox->setBox(orthoBox, false);

    _rstFileOutput->setFilename("default.rst", false);
    _rstFileOutput->write(
        out::Frame{*_simulationBox, thermostat::Thermostat()},
        10,
        100 * fs
    );
    _rstFileOutput->close();

    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    getline(file, line);

    std::istringstream boxLine(line);
    std::string        keyword;
    double             lengthA = 0.0;
    double             lengthB = 0.0;
    double             lengthC = 0.0;
    double             alpha   = 0.0;
    double             beta    = 0.0;
    double             gamma   = 0.0;
    boxLine >> keyword >> lengthA >> lengthB >> lengthC >> alpha >> beta >>
        gamma;

    EXPECT_EQ(keyword, "Box");
    EXPECT_EQ(lengthA, length);
    EXPECT_EQ(lengthB, 0.5 * length);
    EXPECT_EQ(lengthC, 1.0 / 3.0);
    EXPECT_EQ(alpha, 90.0);
    EXPECT_EQ(beta, 90.0);
    EXPECT_EQ(gamma, 90.0);
}

namespace
{
    // values that a fixed number of decimals cannot represent
    const std::vector<double> awkwardValues = {
        1.0 / 3.0,
        -0.1,
        2.0 / 3.0,
        1.0e-20,
        -6.02214076e23,
        1.0e300,
        123456789.123456789,
        std::nextafter(1.0, 2.0),
        std::numeric_limits<double>::min(),
        -0.0,
        -27.44205774028026567635,
    };

    bool sameBits(const double lhs, const double rhs)
    {
        return std::bit_cast<std::uint64_t>(lhs) ==
               std::bit_cast<std::uint64_t>(rhs);
    }

    std::vector<std::string> fieldsOf(const std::string &line)
    {
        std::istringstream       stream(line);
        std::vector<std::string> fields;
        std::string              field;

        while (stream >> field)
        {
            fields.push_back(field);
        }

        return fields;
    }
}   // namespace

/**
 * @brief positions, velocities and forces of the restart file read back as
 * exactly the same doubles (to the last bit, also -0.0)
 *
 * @details a restart is only an exact continuation if no digit is lost: 8
 * decimals give 1e-8 Angstrom per atom, which the forces of a stiff potential
 * amplify
 */
TEST_F(TestRstFileOutput, atomValuesAreWrittenExactly)
{
    std::vector<std::shared_ptr<molsys::Atom>> atoms;
    for (auto &molecule : _simulationBox->getMolecules())
    {
        for (auto &atom : molecule.getAtoms())
        {
            atoms.push_back(atom);
        }
    }

    ASSERT_EQ(atoms.size(), 3);

    for (size_t index = 0; index < 3; ++index)
    {
        const auto shift = 3 * index;
        const auto value = [shift](const size_t axis)
        { return awkwardValues.at((shift + axis) % awkwardValues.size()); };

        atoms.at(index)->setPosition({value(0), value(1), value(2)});
        atoms.at(index)->setVelocity({value(3), value(4), value(5)});
        atoms.at(index)->setForce({value(6), value(7), value(8)});
    }

    _rstFileOutput->setFilename("default.rst", false);
    _rstFileOutput->write(
        out::Frame{*_simulationBox, thermostat::Thermostat()},
        10,
        100 * fs
    );
    _rstFileOutput->close();

    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    getline(file, line);

    for (size_t index = 0; index < 3; ++index)
    {
        getline(file, line);
        const auto fields = fieldsOf(line);
        ASSERT_EQ(fields.size(), 12) << line;

        const auto shift = 3 * index;

        for (size_t column = 0; column < 9; ++column)
        {
            const auto expected =
                awkwardValues.at((shift + column) % awkwardValues.size());
            const auto written = std::stod(fields.at(3 + column));

            EXPECT_TRUE(sameBits(expected, written))
                << "atom " << index << " column " << column << ": "
                << fields.at(3 + column);
        }
    }
}

/**
 * @brief the chi and zeta of a Nose-Hoover chain are written exactly: a chain
 * state with 6 digits changes the continuation of a restarted run
 */
TEST_F(TestRstFileOutput, noseHooverChainIsWrittenExactly)
{
    NoseHooverSettings noseHooverSettings;
    noseHooverSettings.setNoseHooverCouplingFrequencyByUser(1.0e13 * Hz);
    noseHooverSettings.setChiByUser(
        {{0, awkwardValues.at(0)},
         {1, awkwardValues.at(3)},
         {2, awkwardValues.at(4)},
         {3, 0.0}}
    );
    noseHooverSettings.setZetaByUser(
        {{0, awkwardValues.at(1)},
         {1, awkwardValues.at(6)},
         {2, awkwardValues.at(7)},
         {3, 0.0}}
    );
    noseHooverSettings.setNoseHooverChainLengthByUser(3);
    auto thermostat = thermostat::NoseHooverThermostat(noseHooverSettings);

    _rstFileOutput->setFilename("default.rst", false);
    _rstFileOutput
        ->write(out::Frame{*_simulationBox, thermostat}, 10, 100 * fs);
    _rstFileOutput->close();

    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    getline(file, line);

    const std::vector<std::pair<double, double>> expected = {
        {awkwardValues.at(0), awkwardValues.at(1)},
        {awkwardValues.at(3), awkwardValues.at(6)},
        {awkwardValues.at(4), awkwardValues.at(7)},
    };

    for (size_t level = 0; level < expected.size(); ++level)
    {
        getline(file, line);
        const auto fields = fieldsOf(line);
        ASSERT_EQ(fields.size(), 4) << line;

        EXPECT_EQ(fields.at(0), "chi");
        EXPECT_EQ(std::stoul(fields.at(1)), level + 1);
        EXPECT_TRUE(sameBits(expected.at(level).first, std::stod(fields.at(2))))
            << fields.at(2);
        EXPECT_TRUE(sameBits(expected.at(level).second, std::stod(fields.at(3)))
        ) << fields.at(3);
    }
}

/**
 * @brief a subnormal value (below 2.2e-308) is written as a zero of the same
 * sign: the restart reader parses with std::stod, which rejects subnormal
 * numbers, and a file that cannot be read back is worse than a zero
 */
TEST_F(TestRstFileOutput, subnormalValuesAreWrittenAsZero)
{
    constexpr auto tiny = std::numeric_limits<double>::denorm_min();

    std::vector<std::shared_ptr<molsys::Atom>> atoms;
    for (auto &molecule : _simulationBox->getMolecules())
    {
        for (auto &atom : molecule.getAtoms())
        {
            atoms.push_back(atom);
        }
    }

    ASSERT_EQ(atoms.size(), 3);

    atoms.at(0)->setPosition({1.0e-310, -1.0e-310, tiny});
    atoms.at(0)->setVelocity({-tiny, 2.5e-320, 1.0});
    atoms.at(0)->setForce({1.0e-309, -2.0e-309, 0.5});

    _rstFileOutput->setFilename("default.rst", false);
    _rstFileOutput->write(
        out::Frame{*_simulationBox, thermostat::Thermostat()},
        10,
        100 * fs
    );
    _rstFileOutput->close();

    std::ifstream file("default.rst");
    std::string   line;
    getline(file, line);
    getline(file, line);
    getline(file, line);

    const auto fields = fieldsOf(line);
    ASSERT_EQ(fields.size(), 12) << line;

    const std::vector<double> expected =
        {0.0, -0.0, 0.0, -0.0, 0.0, 1.0, 0.0, -0.0, 0.5};

    for (size_t column = 0; column < expected.size(); ++column)
    {
        double written = 1.0;
        EXPECT_NO_THROW(written = std::stod(fields.at(3 + column)))
            << fields.at(3 + column);
        EXPECT_TRUE(sameBits(expected.at(column), written))
            << "column " << column << ": " << fields.at(3 + column);
    }
}
