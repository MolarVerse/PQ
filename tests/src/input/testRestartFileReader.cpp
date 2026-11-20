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

#include "testRestartFileReader.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "moldescriptorReader.hpp"
#include "restartFileReader.hpp"
#include "restartFileSection.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

/**
 * @brief tests determineSection base on the first element of the line
 *
 */
TEST_F(TestRstFileReader, determineSection)
{
    std::string filename = "examples/setup/h2o_qmcfc.rst";
    input::restartFile::RestartFileReader rstFileReader(
        filename,
        *_engine,
        _settings.noseHooverSettings,
        _settings.timings
    );

    auto  lineElements = std::vector<std::string>{"sTeP", "1"};
    auto *section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "step");

    lineElements = std::vector<std::string>{"Box"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "box");

    lineElements = std::vector<std::string>{"notAHeaderSection"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "");
}

/**
 * @brief test full read restart file function
 *
 */
TEST_F(TestRstFileReader, rstFileReading)
{
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );
    input::molDescriptor::MoldescriptorReader moldescriptor(*_engine);

    std::string filename = "examples/setup/h2o-qmcf.rst";
    _settings.fileSettings.setStartFileNameByUser(filename);

    moldescriptor.read();
    ASSERT_NO_THROW(
        input::restartFile::readRestartFile(
            *_engine,
            _settings.noseHooverSettings,
            _settings.timings
        )
    );
}

/**
 * @brief tests that reading an MD restart file writes the time step, the
 * starting step count and the number of steps to the log file
 *
 */
TEST_F(TestRstFileReader, writeStepInfoMD)
{
    settings::GeneralSettings::setJobtype(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );
    _settings.fileSettings.setStartFileNameByUser(
        "examples/setup/h2o-qmcf.rst"
    );
    _engine->getLogOutput().setFilename("default.log", false);

    input::molDescriptor::MoldescriptorReader(*_engine).read();
    input::restartFile::readRestartFile(
        *_engine,
        _settings.noseHooverSettings,
        _settings.timings
    );
    _engine->getLogOutput().close();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);   // "Start File" message
    getline(file, line);
    EXPECT_EQ(line, "         Time Step:           0.5 fs");
    getline(file, line);
    EXPECT_EQ(line, "         Starting Step Count: 1232");
    getline(file, line);
    EXPECT_EQ(line, "         Number of steps:     100");
    getline(file, line);
    EXPECT_EQ(line, "");

    file.close();
    std::remove("default.log");
}

/**
 * @brief tests that reading an optimizer restart file writes the starting
 * step count and the maximum number of optimization steps to the log file
 *
 */
TEST_F(TestRstFileReader, writeStepInfoOpt)
{
    settings::GeneralSettings::setJobtype(JobType::MM_OPT);
    _settings.timings.setNumberOfStepsByUser(100);
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );
    _settings.fileSettings.setStartFileNameByUser(
        "examples/setup/h2o-qmcf.rst"
    );
    _engine->getLogOutput().setFilename("default.log", false);

    input::molDescriptor::MoldescriptorReader(*_engine).read();
    input::restartFile::readRestartFile(
        *_engine,
        _settings.noseHooverSettings,
        _settings.timings
    );
    _engine->getLogOutput().close();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);   // "Start File" message
    getline(file, line);
    EXPECT_EQ(line, "         Starting Step Count:                  1232");
    getline(file, line);
    EXPECT_EQ(line, "         Maximum Number of optimization steps: 100");
    getline(file, line);
    EXPECT_EQ(line, "");

    file.close();
    std::remove("default.log");
    settings::GeneralSettings::setJobtype(JobType::MM_MD);
}
