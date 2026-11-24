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

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "moldescriptorReader.hpp"
#include "restartFileReader.hpp"
#include "restartFileSection.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"
#include "version.hpp"

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
        _settings.timings,
        _settings.generalSettings
    );

    auto  lineElements = std::vector<std::string>{"sTeP", "1"};
    auto *section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "step");

    lineElements = std::vector<std::string>{"Box"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "box");

    lineElements = std::vector<std::string>{"!META_DATA", "version", "v1.0.0"};
    section      = rstFileReader.determineSection(lineElements);
    EXPECT_EQ(section->keyword(), "!meta_data");

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
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);
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
            _settings.timings,
            _settings.generalSettings
        )
    );
}

/**
 * @brief a start file without atoms is refused, whatever else it contains
 *
 */
TEST_F(TestRstFileReader, startFileWithoutAtomsIsRefused)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);

    const std::string filename = "start_file_without_atoms.rst";

    for (const char *content :
         {"", "# only a comment\n\n", "Step 0\nBox 20.0 20.0 20.0\n"})
    {
        {
            std::ofstream file(filename);
            file << content;
        }

        _settings.fileSettings.setStartFileNameByUser(filename);

        ASSERT_THROW_MSG(
            input::restartFile::readRestartFile(
                *_engine,
                _settings.noseHooverSettings,
                _settings.timings
            ),
            exc::RstFileException,
            "Start file \"start_file_without_atoms.rst\" contains no atoms - a "
            "simulation needs at least one atom"
        );
    }

    std::filesystem::remove(filename);
}

/**
 * @brief a start file without atoms is refused, whatever else it contains
 *
 */
TEST_F(TestRstFileReader, startFileWithoutAtomsIsRefused)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);

    const std::string filename = "start_file_without_atoms.rst";

    for (const char *content :
         {"", "# only a comment\n\n", "Step 0\nBox 20.0 20.0 20.0\n"})
    {
        {
            std::ofstream file(filename);
            file << content;
        }

        _settings.fileSettings.setStartFileNameByUser(filename);

        ASSERT_THROW_MSG(
            input::restartFile::readRestartFile(
                *_engine,
                _settings.noseHooverSettings,
                _settings.timings
            ),
            exc::RstFileException,
            "Start file \"start_file_without_atoms.rst\" contains no atoms - a "
            "simulation needs at least one atom"
        );
    }

    std::filesystem::remove(filename);
}

/**
 * @brief tests that reading an MD restart file writes the time step, the
 * starting step count and the number of steps to the log file
 *
 */
TEST_F(TestRstFileReader, writeStepInfoMD)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);
    _settings.timings.setStartingTimeByUser(2 * ps);
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );
    _settings.fileSettings.setStartFileNameByUser(
        "examples/setup/h2o-qmcf.rst"
    );
    const auto _ = std::remove("default.log");
    _engine->getLogOutput().setFilename("default.log", false);

    input::molDescriptor::MoldescriptorReader(*_engine).read();
    input::restartFile::readRestartFile(
        *_engine,
        _settings.noseHooverSettings,
        _settings.timings,
        _settings.generalSettings
    );
    _engine->getLogOutput().close();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);   // "Start File" message
    getline(file, line);   // empty line after it
    getline(file, line);
    EXPECT_EQ(line, "         Time Step:              0.5 fs");
    getline(file, line);
    EXPECT_EQ(line, "         Starting Step Count:    1232");
    getline(file, line);
    EXPECT_EQ(line, "         Number of steps:        100");
    getline(file, line);
    EXPECT_EQ(line, "         Starting Time:          2 ps");
    getline(file, line);
    EXPECT_EQ(line, "         Simulation Time of Run: 0.05 ps");
    getline(file, line);
    EXPECT_EQ(line, "         Total Simulation Time:  2.05 ps");
    getline(file, line);
    EXPECT_EQ(line, "");

    file.close();
    ASSERT_TRUE(std::remove("default.log") == 0);
}

/**
 * @brief tests that reading an optimizer restart file writes the starting
 * step count and the maximum number of optimization steps to the log file
 *
 */
TEST_F(TestRstFileReader, writeStepInfoOpt)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_OPT);
    _settings.timings.setNumberOfStepsByUser(100);
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );
    _settings.fileSettings.setStartFileNameByUser(
        "examples/setup/h2o-qmcf.rst"
    );
    const auto _ = std::remove("default.log");
    _engine->getLogOutput().setFilename("default.log", false);

    input::molDescriptor::MoldescriptorReader(*_engine).read();
    input::restartFile::readRestartFile(
        *_engine,
        _settings.noseHooverSettings,
        _settings.timings,
        _settings.generalSettings
    );
    _engine->getLogOutput().close();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);   // "Start File" message
    getline(file, line);   // empty line after it
    getline(file, line);
    EXPECT_EQ(line, "         Starting Step Count:                  1232");
    getline(file, line);
    EXPECT_EQ(line, "         Maximum Number of optimization steps: 100");
    getline(file, line);
    EXPECT_EQ(line, "");

    file.close();
    ASSERT_TRUE(std::remove("default.log") == 0);
}

namespace
{
    // copies the example restart file and prepends the given lines
    void writeRstFileWithHeader(
        const std::string &filename,
        const std::string &header
    )
    {
        std::ifstream source("examples/setup/h2o-qmcf.rst");
        std::ofstream target(filename);
        target << header << source.rdbuf();
    }
}   // namespace

/**
 * @brief the metadata line of a restart file is read into the settings
 *
 */
TEST_F(TestRstFileReader, readsMetaData)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );

    const std::string filename = "metaDataTest.rst";
    writeRstFileWithHeader(filename, "!META_DATA version v1.2.3\n");
    _settings.fileSettings.setStartFileNameByUser(filename);

    input::molDescriptor::MoldescriptorReader(*_engine).read();

    ASSERT_NO_THROW(
        input::restartFile::readRestartFile(
            *_engine,
            _settings.noseHooverSettings,
            _settings.timings,
            _settings.generalSettings
        )
    );

    EXPECT_EQ(
        _settings.generalSettings.getRestartVersion(),
        SemVer::from_string("v1.2.3")
    );
    EXPECT_EQ(_settings.timings.getStartingStepCount(), 1232);

    std::remove(filename.c_str());
}

/**
 * @brief a restart file without metadata leaves the restart version invalid
 *
 */
TEST_F(TestRstFileReader, noMetaDataKeepsVersionInvalid)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );
    _settings.fileSettings.setStartFileNameByUser(
        "examples/setup/h2o-qmcf.rst"
    );

    input::molDescriptor::MoldescriptorReader(*_engine).read();
    input::restartFile::readRestartFile(
        *_engine,
        _settings.noseHooverSettings,
        _settings.timings,
        _settings.generalSettings
    );

    EXPECT_FALSE(isValid(_settings.generalSettings.getRestartVersion()));
}

/**
 * @brief a malformed metadata line makes reading the restart file fail
 *
 */
TEST_F(TestRstFileReader, malformedMetaDataThrows)
{
    _settings.generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings.timings.setTimeStepByUser(0.5 * fs);
    _settings.timings.setNumberOfStepsByUser(100);
    _settings.fileSettings.setMolDescriptorFileNameByUser(
        "examples/setup/moldescriptor.dat"
    );

    const std::string filename = "malformedMetaDataTest.rst";
    writeRstFileWithHeader(filename, "!META_DATA colour red\n");
    _settings.fileSettings.setStartFileNameByUser(filename);

    input::molDescriptor::MoldescriptorReader(*_engine).read();

    EXPECT_THROW_MSG(
        input::restartFile::readRestartFile(
            *_engine,
            _settings.noseHooverSettings,
            _settings.timings,
            _settings.generalSettings
        ),
        exc::RstFileException,
        "Unrecognized metadata key 'colour' at line 1"
    );

    std::remove(filename.c_str());
}
