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
#include <string>

#include "executablePath.hpp"
#include "tempDirTest.hpp"

namespace
{
    constexpr auto _programName = "pq-test-program";

#if defined(_WIN32)
    constexpr auto _separator = ";";
#else
    constexpr auto _separator = ":";
#endif

    using SearchPathTest = TempDirTest;
}   // namespace

TEST(ExecutablePathTest, resolvesRunningExecutable)
{
    const auto executable = utilities::executablePath();

    EXPECT_FALSE(executable.empty());
    EXPECT_TRUE(std::filesystem::is_regular_file(executable));
    EXPECT_EQ(executable.stem(), "testExecutablePath");
}

TEST(ExecutablePathTest, resolvesInstalledDataRelativeToExecutable)
{
    const auto executable = utilities::executablePath();

    EXPECT_EQ(
        utilities::installedDataPath("references"),
        executable.parent_path().parent_path() / "share" / "PQ" / "references"
    );
}

TEST_F(SearchPathTest, isExecutableFileNeedsRegularExecutableFile)
{
    EXPECT_TRUE(utilities::isExecutableFile(_makeFile(_root / "run", true)));
    EXPECT_FALSE(utilities::isExecutableFile(_makeFile(_root / "read", false)));
    EXPECT_FALSE(utilities::isExecutableFile(_makeDir("dir")));
    EXPECT_FALSE(utilities::isExecutableFile(_root / "missing"));
}

TEST_F(SearchPathTest, firstExecutableWins)
{
    const auto empty  = _makeDir("empty");
    const auto first  = _makeDir("first");
    const auto second = _makeDir("second");
    _makeFile(first / _programName, true);
    _makeFile(second / _programName, true);

    const auto searchPath = empty.string() + _separator + _separator +
                            first.string() + _separator + second.string();

    EXPECT_EQ(
        utilities::findInSearchPath(searchPath, _programName),
        first / _programName
    );
}

TEST_F(SearchPathTest, skipsEntriesThatCannotRun)
{
    const auto noExec = _makeDir("noExec");
    const auto isDir  = _makeDir("isDir");
    const auto good   = _makeDir("good");
    _makeFile(noExec / _programName, false);
    std::filesystem::create_directories(isDir / _programName);
    _makeFile(good / _programName, true);

    const auto searchPath = isDir.string() + _separator + noExec.string() +
                            _separator + good.string() + _separator;

    EXPECT_EQ(
        utilities::findInSearchPath(searchPath, _programName),
        good / _programName
    );
}

TEST_F(SearchPathTest, nothingFound)
{
    _makeFile(_makeDir("noExec") / _programName, false);

    EXPECT_FALSE(utilities::findInSearchPath("", _programName).has_value());
    EXPECT_FALSE(
        utilities::findInSearchPath(_separator, _programName).has_value()
    );
    EXPECT_FALSE(
        utilities::findInSearchPath((_root / "noExec").string(), _programName)
            .has_value()
    );
}
