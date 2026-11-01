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
    )
    {
        return settings._determineMostCommonPrefix();
    }

    static void replaceDefaultValues(
        OutputFileSettings& settings,
        const std::string&  prefix
    )
    {
        settings._replaceDefaultValues(prefix);
    }
};

/**
 * @brief tests setting output frequency
 *
 */
TEST(TestOutputSettings, setSpecialOutputFrequency)
{
    OutputFileSettings settings;
    settings.setOutputFrequency(0);
    EXPECT_EQ(settings.getOutputFrequency(), UINT64_MAX);
}

/**
 * @brief tests that output metadata is opt-in
 *
 */
TEST(TestOutputSettings, includeOutputMetadataDefaultsToFalse)
{
    OutputFileSettings settings;
    EXPECT_FALSE(settings.getIncludeOutputMetadata());

    settings.setIncludeOutputMetadata(true);
    EXPECT_TRUE(settings.getIncludeOutputMetadata());

    settings.setIncludeOutputMetadata(false);
}

TEST(TestOutputSettings, determinesMostCommonConfiguredPrefix)
{
    OutputFileSettings settings;
    settings.setRestartFileName("production.rst");
    settings.setEnergyFileName("production.en");
    settings.setTrajectoryFileName("production.xyz");
    settings.setOptFileName("optimization.opt");
    settings.setFilePrefix("production");
    settings.setOverwriteOutputFiles(true);
    settings.setOutputFrequency(5);

    EXPECT_EQ(
        TestOutputFileSettings::determineMostCommonPrefix(settings),
        "production"
    );
    EXPECT_EQ(settings.getOptFileName(), "optimization.opt");
    EXPECT_EQ(settings.getFilePrefix(), "production");
    EXPECT_TRUE(settings.isFilePrefixSet());
    EXPECT_TRUE(settings.getOverwriteOutputFiles());
    EXPECT_EQ(settings.getOutputFrequency(), 5);
}

TEST(TestOutputSettings, setsHybridCenterFilename)
{
    OutputFileSettings settings;
    settings.setHybridCenterFileName("center.xyz");
    EXPECT_EQ(settings.getHybridCenterFileName(), "center.xyz");
}

TEST(TestOutputSettings, replacesDefaultHybridCenterFilenameWithPrefix)
{
    OutputFileSettings settings;
    settings.setHybridCenterFileName(DefaultFiles::hybridCenterFile);
    TestOutputFileSettings::replaceDefaultValues(settings, "centerPrefix");

    EXPECT_EQ(settings.getHybridCenterFileName(), "centerPrefix.center.xyz");
}
