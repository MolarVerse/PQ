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

#include "testEnergyOutput.hpp"

#include <cstddef>
#include <format>
#include <fstream>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>

#include "constraintSettings.hpp"
#include "energyColumns.hpp"
#include "forceFieldSettings.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "systemInfo.hpp"
#include "thermostatSettings.hpp"
#include "units.hpp"
#include "vector3d.hpp"

/**
 * @brief tests writing timestep metadata
 *
 */
TEST_F(TestEnergyOutput, writeHeader)
{
    _energyOutput->setFilename("default.en", false);
    _energyOutput->writeHeader(0.5 * fs);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, out::energyFileMarker());
    std::getline(file, line);
    EXPECT_EQ(line, "# timestep = 0.5 fs");
}

/**
 * @brief tests the marker and column names lines before the first data line
 *
 */
TEST_F(TestEnergyOutput, columnHeader)
{
    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(1, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, std::format("# PQ energy file {}", sysinfo::VERSION));
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "#SIMULATION-TIME[step] TEMPERATURE[K] PRESSURE[bar] E(TOT)[kcal/mol] "
        "E(KIN)[kcal/mol] E(INTRA)[kcal/mol] E(COUL)[kcal/mol] "
        "E(NON-COUL)[kcal/mol] MOMENTUM[amuA/fs] LOOPTIME[s]"
    );
}

/**
 * @brief tests the line order with timestep metadata and that the header
 * lines are written only once
 *
 */
TEST_F(TestEnergyOutput, columnHeaderOnceAfterMetadata)
{
    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->writeHeader(0.5 * fs);
    _energyOutput->write(1, *_physicalData, *_settings);
    _energyOutput->write(2, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream            file("default.en");
    std::vector<std::string> lines;
    for (std::string line; std::getline(file, line);)
    {
        lines.push_back(line);
    }

    ASSERT_EQ(lines.size(), 5);
    EXPECT_EQ(lines[0], out::energyFileMarker());
    EXPECT_EQ(lines[1], "# timestep = 0.5 fs");
    EXPECT_TRUE(lines[2].starts_with("#SIMULATION-TIME[step] TEMPERATURE[K] "));
    EXPECT_TRUE(lines[3].starts_with("         1\t"));
    EXPECT_TRUE(lines[4].starts_with("         2\t"));
}

/**
 * @brief tests that the header has one token per data column, in the order
 * in which PQAnalysis numbers the info file cells, with every optional column
 * active
 *
 */
TEST_F(TestEnergyOutput, columnHeaderMatchesDataAndInfo)
{
    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::ON);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::Z);
    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::NOSE_HOOVER
    );
    _settings->constraintSettings.activateDistanceConstraintsByUser();
    _settings->generalSettings.setJobtypeByUser(JobType::QMMM_MD);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(1, *_physicalData, *_settings);
    _energyOutput->close();

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(1.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    const auto splitWords = [](const std::string &line)
    {
        std::vector<std::string> words;
        std::stringstream        stream(line);
        for (std::string word; stream >> word;)
        {
            words.push_back(word);
        }
        return words;
    };

    std::ifstream enFile("default.en");
    std::string   header;
    std::string   data;
    std::getline(enFile, header);   // marker
    std::getline(enFile, header);
    std::getline(enFile, data);

    ASSERT_TRUE(header.starts_with("#SIMULATION-TIME[step] "));
    const auto tokens = splitWords(header.substr(1));

    EXPECT_EQ(tokens.size(), splitWords(data).size());
    EXPECT_EQ(tokens.size(), out::EnergyColumns::maxColumns + 1);

    // info cells as PQAnalysis InfoFileReader reads them
    std::ifstream            infoFile("default.info");
    std::vector<std::string> infoLabels;
    std::string              line;
    for (auto lineIndex = 0; std::getline(infoFile, line); ++lineIndex)
    {
        if (lineIndex < 3 || line.empty())
        {
            continue;
        }
        if (line.starts_with("-"))
        {
            break;
        }

        const auto words = splitWords(line);
        infoLabels.push_back(std::format("{}[{}]", words[1], words[3]));
        if (words.size() == 8)
        {
            infoLabels.push_back(std::format("{}[{}]", words[4], words[6]));
        }
    }

    ASSERT_EQ(tokens.size(), infoLabels.size());
    EXPECT_EQ(tokens[0], "SIMULATION-TIME[step]");
    EXPECT_EQ(infoLabels[0], "SIMULATION-TIME[ps]");

    for (std::size_t index = 1; index < tokens.size(); ++index)
    {
        EXPECT_EQ(tokens[index], infoLabels[index]);
    }
}

/**
 * @brief tests writing energy output file
 *
 * @details minimal output data
 *
 */
TEST_F(TestEnergyOutput, forceFieldNotActive)
{
    _physicalData->setTemperature(toTemperature(1.0));
    _physicalData->setPressure(2.0 * bar);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linalg::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);
    _physicalData->setLoopTime(0.1);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(100.0, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);   // marker
    std::getline(file, line);   // column names
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100\t      1.000000000000\t      2.000000000000\t     "
        "12.000000000000\t      3.000000000000\t     "
        "19.000000000000\t      4.000000000000\t      5.000000000000\t         "
        "1.03923e+01\t     0.10000"
    );
}

/**
 * @brief tests writing energy output file
 *
 * @details force field is set
 *
 */
TEST_F(TestEnergyOutput, forceFieldActive)
{
    _physicalData->setTemperature(toTemperature(1.0));
    _physicalData->setPressure(2.0 * bar);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linalg::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    _physicalData->setBondEnergy(19.0);
    _physicalData->setAngleEnergy(20.0);
    _physicalData->setDihedralEnergy(21.0);
    _physicalData->setImproperEnergy(22.0);
    _physicalData->setLoopTime(0.1);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::ON);
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(100.0, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);   // marker
    std::getline(file, line);   // column names
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100\t      1.000000000000\t      2.000000000000\t     "
        "94.000000000000\t      3.000000000000\t     "
        "19.000000000000\t      4.000000000000\t      5.000000000000\t     "
        "19.000000000000\t     20.000000000000\t     "
        "21.000000000000\t     22.000000000000\t         1.03923e+01\t     "
        "0.10000"
    );
}

/**
 * @brief tests writing energy output file
 *
 * @details manostat is set
 *
 */
TEST_F(TestEnergyOutput, manostatActive)
{
    _physicalData->setTemperature(toTemperature(1.0));
    _physicalData->setPressure(2.0 * bar);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linalg::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    _physicalData->setVolume(19.0);
    _physicalData->setDensity(20.0);
    _physicalData->setLoopTime(0.1);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(100.0, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);   // marker
    std::getline(file, line);   // column names
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100\t      1.000000000000\t      2.000000000000\t     "
        "12.000000000000\t      3.000000000000\t     "
        "19.000000000000\t      4.000000000000\t      5.000000000000\t     "
        "19.000000000000\t     20.000000000000\t   "
        "      1.03923e+01\t     0.10000"
    );
}

/**
 * @brief tests writing energy output file with fixed axis
 *
 * @details manostat and fixed_axis are set
 *
 */
TEST_F(TestEnergyOutput, manostatActiveWithFixedAxis)
{
    _physicalData->setTemperature(toTemperature(1.0));
    _physicalData->setPressure(2.0 * bar);
    _physicalData->setCoupledPressure(2.5 * bar);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linalg::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    _physicalData->setVolume(19.0);
    _physicalData->setDensity(20.0);
    _physicalData->setLoopTime(0.1);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::Z);
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(100.0, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);   // marker
    std::getline(file, line);   // column names
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100\t      1.000000000000\t      2.000000000000\t     "
        "12.000000000000\t      3.000000000000\t     "
        "19.000000000000\t      4.000000000000\t      5.000000000000\t     "
        "19.000000000000\t     20.000000000000\t      2.500000000000\t   "
        "      1.03923e+01\t     0.10000"
    );
}

/**
 * @brief tests writing energy output file
 *
 * @details qm is active is set
 *
 */
TEST_F(TestEnergyOutput, qmActive)
{
    _physicalData->reset();

    _physicalData->setTemperature(toTemperature(1.0));
    _physicalData->setPressure(2.0 * bar);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setMomentum(linalg::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(0.0);
    _physicalData->setIntraNonCoulombEnergy(0.0);

    _physicalData->setQMEnergy(5.0);

    _physicalData->setVolume(19.0);
    _physicalData->setDensity(20.0);
    _physicalData->setLoopTime(0.1);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->generalSettings.setJobtypeByUser(JobType::QM_MD);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(100.0, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);   // marker
    std::getline(file, line);   // column names
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100\t      1.000000000000\t      2.000000000000\t      "
        "8.000000000000\t      5.000000000000\t      "
        "0.000000000000\t      3.000000000000\t      0.000000000000\t         "
        "1.03923e+01\t     0.10000"
    );
}

/**
 * @brief tests writing energy output file
 *
 * @details nose hoover is active
 *
 */
TEST_F(TestEnergyOutput, noseHooverActive)
{
    _physicalData->setTemperature(toTemperature(1.0));
    _physicalData->setPressure(2.0 * bar);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linalg::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);
    _physicalData->setNoseHooverMomentumEnergy(11.0);
    _physicalData->setNoseHooverFrictionEnergy(12.0);
    _physicalData->setLoopTime(0.1);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);
    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::NOSE_HOOVER
    );

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(100.0, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);   // marker
    std::getline(file, line);   // column names
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "       100\t      1.000000000000\t      2.000000000000\t     "
        "12.000000000000\t      3.000000000000\t     "
        "19.000000000000\t      4.000000000000\t      5.000000000000\t     "
        "11.000000000000\t     12.000000000000\t         "
        "1.03923e+01\t     0.10000"
    );
}

/**
 * @brief tests writing energy output file
 *
 * @details qmmm is active: E(QM) is corrected by the QM reference energy,
 * which is printed after N(SM-MOL)
 *
 */
TEST_F(TestEnergyOutput, qmmmActiveSubtractsQMReferenceEnergy)
{
    _physicalData->reset();

    _physicalData->setTemperature(toTemperature(1.0));
    _physicalData->setPressure(2.0 * bar);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linalg::Vec3D(6.0));
    _physicalData->setQMEnergy(7.0);
    _physicalData->setQMReferenceEnergy(2.0);
    _physicalData->setNumberOfSmoothingMolecules(1.0);
    _physicalData->setLoopTime(0.1);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->generalSettings.setJobtypeByUser(JobType::QMMM_MD);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);
    _settings->thermostatSettings.setThermostatTypeByUser(ThermostatType::NONE);

    _energyOutput->setFilename("default.en", false);
    _energyOutput->write(100.0, *_physicalData, *_settings);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);   // marker
    std::getline(file, line);   // column names
    std::getline(file, line);

    std::vector<double> fields;
    std::stringstream   stream(line);
    for (std::string field; std::getline(stream, field, '\t');)
        fields.push_back(std::stod(field));

    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);

    ASSERT_EQ(fields.size(), 14);
    EXPECT_DOUBLE_EQ(fields[3], 17.0);   // E(TOT)
    EXPECT_DOUBLE_EQ(fields[4], 5.0);    // E(QM)
    EXPECT_DOUBLE_EQ(fields[10], 1.0);   // N(SM-MOL)
    EXPECT_DOUBLE_EQ(fields[11], 2.0);   // E(QM-REF)
}
