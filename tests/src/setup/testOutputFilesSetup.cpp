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

#include <cstdio>
#include <string>
#include <vector>

#include "mdEngine.hpp"
#include "optEngine.hpp"
#include "outputFileSettings.hpp"
#include "outputFilesSetup.hpp"
#include "settings.hpp"
#include "testSetup.hpp"

using namespace setup;
using namespace settings;

namespace
{
    // OutputFileSettings caches filenames after their first replace from the
    // default value, so we must use the same prefix across tests. Each test
    // wipes the on-disk files first so opening doesn't fail with
    // "file already exists".
    constexpr const char *_PREFIX = "ofsTest";

    void cleanupPrefix()
    {
        const std::vector<std::string> suffixes = {
            ".log",      ".info",     ".rst",        ".en",        ".xyz",
            ".timings",  ".force",    ".instant_en", ".vel",       ".chrg",
            ".mom",      ".vir",      ".stress",     ".box",       ".rpmd.rst",
            ".rpmd.xyz", ".rpmd.vel", ".rpmd.force", ".rpmd.chrg", ".rpmd.en",
            ".opt",      ".ref"
        };
        for (const auto &s : suffixes)
            ::remove((std::string(_PREFIX) + s).c_str());
    }
}   // namespace

TEST_F(TestSetup, setupOutputFilesOptJobReplaceDefaultsAndAssignsOptFile)
{
    cleanupPrefix();
    Settings::setJobtype(JobType::MM_OPT);
    Settings::setIsRingPolymerMDActivated(false);
    OutputFileSettings::setFilePrefix(_PREFIX);

    EXPECT_NO_THROW(setupOutputFiles(*_engine));

    // After setup, the log/timings/info filenames are now prefix-substituted.
    EXPECT_EQ(
        OutputFileSettings::getLogFileName(),
        std::string(_PREFIX) + ".log"
    );
    EXPECT_EQ(
        OutputFileSettings::getOptFileName(),
        std::string(_PREFIX) + ".opt"
    );

    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesMDPathRunsWithoutThrowing)
{
    cleanupPrefix();
    Settings::setJobtype(JobType::MM_MD);
    Settings::setIsRingPolymerMDActivated(false);
    OutputFileSettings::setFilePrefix(_PREFIX);

    OutputFilesSetup s(*_mdEngine);
    EXPECT_NO_THROW(s.setup());

    cleanupPrefix();
}

TEST_F(TestSetup, setupOutputFilesRPMDPathRunsWithoutThrowing)
{
    cleanupPrefix();
    Settings::setJobtype(JobType::MM_MD);
    Settings::setIsRingPolymerMDActivated(true);
    OutputFileSettings::setFilePrefix(_PREFIX);

    OutputFilesSetup s(*_mdEngine);
    EXPECT_NO_THROW(s.setup());

    Settings::setIsRingPolymerMDActivated(false);
    cleanupPrefix();
}
