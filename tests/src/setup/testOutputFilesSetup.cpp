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

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "h5mdOutput.hpp"
#include "outputFileSettings.hpp"
#include "outputFilesSetup.hpp"
#include "setup.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

namespace
{
    // settings::OutputFileSettings caches filenames after their first replace
    // from the default value, so we must use the same prefix across tests. Each
    // test wipes the on-disk files first so opening doesn't fail with "file
    // already exists".
    constexpr const char *PREFIX = "ofsTest";

    void cleanupPrefix()
    {
        const std::vector<std::string> suffixes = {
            ".log",        ".info",       ".rst",        ".en",
            ".xyz",        ".center.xyz", ".timings",    ".force",
            ".instant_en", ".vel",        ".chrg",       ".mom",
            ".vir",        ".stress",     ".box",        ".rpmd.rst",
            ".rpmd.xyz",   ".rpmd.vel",   ".rpmd.force", ".rpmd.chrg",
            ".rpmd.en",    ".opt",        ".ref",        ".extxyz",
            ".h5md"
        };

        for (const auto &suffix : suffixes)
        {
            static_cast<void>(
                std::filesystem::remove((std::string(PREFIX) + suffix).c_str())
            );
        }
    }
}   // namespace

TEST_F(TestSetup, setupOutputFilesOptJobReplaceDefaultsAndAssignsOptFile)
{
    cleanupPrefix();
    _settings->generalSettings.setJobtypeByUser(JobType::MM_OPT);
    _settings->outputFiles.setFilePrefixByUser(PREFIX);
    _settings->timings.setTimeStepByUser(0.5 * fs);
    _settings->timings.setNumberOfStepsByUser(100U);
    _settings->finalize();

    EXPECT_NO_THROW(setup::setupOutputFiles(*_engine));

    // After setup, the log/timings/info filenames are now prefix-substituted.
    EXPECT_EQ(
        _settings->outputFiles.getLogFileName(),
        std::string(PREFIX) + ".log"
    );
    EXPECT_EQ(
        _settings->outputFiles.getOptFileName(),
        std::string(PREFIX) + ".opt"
    );

    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesHybridPathAssignsCenterFile)
{
    cleanupPrefix();
    _settings->generalSettings.setJobtypeByUser(JobType::QMMM_MD);
    _settings->outputFiles.setFilePrefixByUser(PREFIX);
    _settings->timings.setTimeStepByUser(0.5 * fs);
    _settings->timings.setNumberOfStepsByUser(100U);

    _settings->finalize();
    setup::OutputFilesSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());

    EXPECT_EQ(
        _mdEngine->getXyzHybridCenterOutput().getFilename(),
        std::string(PREFIX) + ".center.xyz"
    );

    _mdEngine->getXyzHybridCenterOutput().close();
    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesMDPathPreservesLegacyEnergyFormatByDefault)
{
    cleanupPrefix();
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings->outputFiles.setFilePrefixByUser(PREFIX);
    _settings->timings.setTimeStepByUser(0.5 * fs);
    _settings->timings.setNumberOfStepsByUser(100U);

    _settings->finalize();
    setup::OutputFilesSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());

    _mdEngine->getEnergyOutput().close();
    _mdEngine->getInstantEnergyOutput().close();

    std::ifstream energyFile(std::string(PREFIX) + ".en");
    std::ifstream instantEnergyFile(std::string(PREFIX) + ".instant_en");
    std::string   line;

    EXPECT_FALSE(static_cast<bool>(std::getline(energyFile, line)));
    EXPECT_FALSE(static_cast<bool>(std::getline(instantEnergyFile, line)));

    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesMDPathWritesEnabledMetadata)
{
    cleanupPrefix();
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings->outputFiles.setFilePrefixByUser(PREFIX);
    _settings->outputFiles.setIncludeOutputMetadataByUser(true);
    _settings->timings.setTimeStepByUser(0.5 * fs);
    _settings->timings.setNumberOfStepsByUser(100U);

    _settings->finalize();
    setup::OutputFilesSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());

    _mdEngine->getEnergyOutput().close();
    _mdEngine->getInstantEnergyOutput().close();

    std::ifstream energyFile(std::string(PREFIX) + ".en");
    std::ifstream instantEnergyFile(std::string(PREFIX) + ".instant_en");
    std::string   line;

    std::getline(energyFile, line);   // marker
    std::getline(energyFile, line);
    EXPECT_EQ(line, "# timestep = 0.5 fs");
    std::getline(instantEnergyFile, line);   // marker
    std::getline(instantEnergyFile, line);
    EXPECT_EQ(line, "# timestep = 0.5 fs");

    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesMDPathExtxyzReplacesTrajectoryFiles)
{
    cleanupPrefix();
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings->outputFiles.setFilePrefixByUser(PREFIX);
    _settings->outputFiles.setTrajectoryFormatByUser(TrajectoryFormat::EXTXYZ);
    _settings->timings.setNumberOfStepsByUser(100U);

    _settings->finalize();
    setup::OutputFilesSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());

    const std::string prefix = PREFIX;

    EXPECT_TRUE(std::filesystem::exists(prefix + ".extxyz"));
    EXPECT_TRUE(std::filesystem::exists(prefix + ".rst"));

    for (const auto *suffix : {".xyz", ".vel", ".force", ".chrg"})
    {
        EXPECT_FALSE(std::filesystem::exists(prefix + suffix)) << suffix;
    }

    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesMDPathH5mdReplacesTrajectoryFiles)
{
    cleanupPrefix();
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings->outputFiles.setFilePrefixByUser(PREFIX);
    _settings->outputFiles.setTrajectoryFormatByUser(TrajectoryFormat::H5MD);
    _settings->timings.setTimeStepByUser(0.5 * fs);
    _settings->timings.setNumberOfStepsByUser(100U);

    _settings->finalize();
    setup::OutputFilesSetup setup(*_mdEngine);

    if (!out::H5mdOutput::available())
    {
        EXPECT_THROW_MSG(
            setup.setup(),
            exc::CompileTimeException,
            "traj_format = h5md needs PQ built with -DBUILD_WITH_HDF5=ON"
        );
        cleanupPrefix();
        return;
    }

    EXPECT_NO_THROW(setup.setup());

    const std::string prefix = PREFIX;
    EXPECT_TRUE(std::filesystem::exists(prefix + ".h5md"));

    for (const auto *suffix : {".xyz", ".vel", ".force", ".chrg"})
    {
        EXPECT_FALSE(std::filesystem::exists(prefix + suffix)) << suffix;
    }

    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesRPMDPathRunsWithoutThrowing)
{
    cleanupPrefix();
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);
    _settings->outputFiles.setFilePrefixByUser(PREFIX);
    _settings->timings.setNumberOfStepsByUser(100U);

    _settings->finalize();
    setup::OutputFilesSetup setup(*_mdEngine);
    EXPECT_NO_THROW(setup.setup());

    cleanupPrefix();
}

/**
 * @brief tests that the engine setup info writes the job type to the log file
 *
 */
TEST_F(TestSetup, writeEngineSetupInfo)
{
    auto &logOutput = _engine->getLogOutput();
    logOutput.setFilename("engineSetupInfo.log", false);

    setup::writeEngineSetupInfo(logOutput, JobType::MM_MD);
    logOutput.close();

    std::ifstream file("engineSetupInfo.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "INFO:    Setup of Engine for jobtype: MM_MD");

    file.close();
    ASSERT_TRUE(std::remove("engineSetupInfo.log") == 0);
}
