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

#ifndef TESTS__INCLUDE__TEST_UTILS__TEMP_DIR_TEST_HPP
#define TESTS__INCLUDE__TEST_UTILS__TEMP_DIR_TEST_HPP

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

/**
 * @brief fixture with a fresh directory per test, removed afterwards
 */
class TempDirTest : public ::testing::Test
{
   protected:
    std::filesystem::path _root;

    void SetUp() override;
    void TearDown() override;

    [[nodiscard]] std::filesystem::path _makeDir(const std::string &name) const;
    static std::filesystem::path        _makeFile(
               const std::filesystem::path &path,
               bool                         executable
           );
};

inline void TempDirTest::SetUp()
{
    const auto *const info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    _root = std::filesystem::temp_directory_path() /
            (std::string("pq-") + info->test_suite_name() + "-" + info->name());
    std::filesystem::remove_all(_root);
    std::filesystem::create_directories(_root);
}

inline void TempDirTest::TearDown() { std::filesystem::remove_all(_root); }

inline std::filesystem::path TempDirTest::_makeDir(const std::string &name
) const
{
    auto path = _root / name;
    std::filesystem::create_directories(path);
    return path;
}

inline std::filesystem::path TempDirTest::_makeFile(
    const std::filesystem::path &path,
    bool                         executable
)
{
    std::ofstream(path) << "#!/bin/sh\n";

    auto perms = std::filesystem::perms::owner_read |
                 std::filesystem::perms::owner_write;
    if (executable)
    {
        perms |= std::filesystem::perms::owner_exec;
    }
    std::filesystem::permissions(path, perms);

    return path;
}

#endif   // TESTS__INCLUDE__TEST_UTILS__TEMP_DIR_TEST_HPP
