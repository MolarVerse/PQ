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

#include <cstdint>

#include "outputFileSettings.hpp"

struct TestOutputFileSettings
{
    static std::string determineMostCommonPrefix(
        const OutputFileSettings& settings
    );

    static void replaceDefaultValues(
        OutputFileSettings& settings,
        const std::string&  prefix
    );
};

std::string TestOutputFileSettings::determineMostCommonPrefix(
    const OutputFileSettings& settings
)
{
    return settings._determineMostCommonPrefix();
}

void TestOutputFileSettings::replaceDefaultValues(
    OutputFileSettings& settings,
    const std::string&  prefix
)
{
    settings._replaceDefaultValues(prefix);
}

/**
 * @brief tests setting output frequency
 *
 */
TEST(TestOutputSettings, setSpecialOutputFrequency)
{
    OutputFileSettings settings;
    settings.setOutputFrequencyByUser(0);
    EXPECT_EQ(settings.get_outputFrequency(), UINT64_MAX);
}

/**
 * @brief tests that output metadata is opt-in
 *
 */
TEST(TestOutputSettings, includeOutputMetadataDefaultsToFalse)
{
    OutputFileSettings settings;
    EXPECT_FALSE(settings.get_includeOutputMetadata());

    settings.set_includeOutputMetadataByUser(true);
    EXPECT_TRUE(settings.get_includeOutputMetadata());

    settings.set_includeOutputMetadataByUser(false);
}

TEST(TestOutputSettings, determinesMostCommonConfiguredPrefix)
{
    OutputFileSettings settings;
    settings.set_rstFileByUser("production.rst");
    settings.set_energyFileByUser("production.en");
    settings.set_trajFileByUser("production.xyz");
    settings.set_optFileByUser("optimization.opt");
    settings.set_filePrefixByUser("production");
    settings.set_overwriteOutputFilesByUser(true);
    settings.setOutputFrequencyByUser(5);

    EXPECT_EQ(
        TestOutputFileSettings::determineMostCommonPrefix(settings),
        "production"
    );
    EXPECT_EQ(settings.get_optFile(), "optimization.opt");
    EXPECT_EQ(settings.get_filePrefix(), "production");
    EXPECT_TRUE(settings.get_overwriteOutputFiles());
    EXPECT_EQ(settings.get_outputFrequency(), 5);
}

TEST(TestOutputSettings, setsHybridCenterFilename)
{
    OutputFileSettings settings;
    settings.set_hybridCenterFileByUser("center.xyz");
    EXPECT_EQ(settings.get_hybridCenterFile(), "center.xyz");
}

TEST(TestOutputSettings, replacesDefaultHybridCenterFilenameWithPrefix)
{
    OutputFileSettings settings;
    TestOutputFileSettings::replaceDefaultValues(settings, "centerPrefix");

    EXPECT_EQ(settings.get_hybridCenterFile(), "centerPrefix.center.xyz");
}

TEST(TestOutputSettings, preservesUserSetHybridCenterFilenameEqualToDefault)
{
    OutputFileSettings settings;
    settings.set_hybridCenterFileByUser(DefaultFiles::hybridCenterFile);
    TestOutputFileSettings::replaceDefaultValues(settings, "centerPrefix");

    EXPECT_EQ(settings.get_hybridCenterFile(), DefaultFiles::hybridCenterFile);
}

TEST(TestOutputSettings, trajectoryFormatDefaultsToXyz)
{
    OutputFileSettings settings;
    EXPECT_EQ(settings.get_trajFormat(), TrajectoryFormat::XYZ);

    settings.set_trajFormatByUser(TrajectoryFormat::EXTXYZ);
    EXPECT_EQ(settings.get_trajFormat(), TrajectoryFormat::EXTXYZ);
}

TEST(TestOutputSettings, extxyzFilenameFollowsPrefix)
{
    OutputFileSettings settings;
    settings.set_filePrefixByUser("run");
    settings.finalize();
    EXPECT_EQ(settings.get_extxyzFile(), "run.extxyz");
    EXPECT_EQ(settings.get_h5mdFile(), "run.h5md");
    EXPECT_EQ(settings.get_aseTrajFile(), "run.ase.traj");
    EXPECT_EQ(settings.get_zarrTrajFile(), "run.traj.zarr");

    OutputFileSettings named;
    named.set_extxyzFileByUser("water.extxyz");
    named.set_h5mdFileByUser("water.h5md");
    named.set_aseTrajFileByUser("water.ase.traj");
    named.set_zarrTrajFileByUser("water.traj.zarr");
    named.finalize();
    EXPECT_EQ(named.get_extxyzFile(), "water.extxyz");
    EXPECT_EQ(named.get_h5mdFile(), "water.h5md");
    EXPECT_EQ(named.get_aseTrajFile(), "water.ase.traj");
    EXPECT_EQ(named.get_zarrTrajFile(), "water.traj.zarr");
    EXPECT_EQ(named.get_rstFile(), "water.rst");
}
