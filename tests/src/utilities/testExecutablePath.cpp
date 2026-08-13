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

#include "executablePath.hpp"

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
