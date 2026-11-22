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
#include <pybind11/embed.h>

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "embeddedPython.hpp"
#include "interrupt.hpp"

namespace
{
    constexpr auto _programName = "python3.99";

    /**
     * @brief fresh directory with fake interpreters, removed afterwards
     */
    class TestEmbeddedPython : public ::testing::Test
    {
       protected:
        std::filesystem::path _root;

        void SetUp() override;
        void TearDown() override;

        [[nodiscard]] std::filesystem::path _makeDir(const std::string &name
        ) const;
        static std::filesystem::path _makeFile(
            const std::filesystem::path &path,
            bool                         executable
        );
    };

    void TestEmbeddedPython::SetUp()
    {
        const auto *const info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        _root = std::filesystem::temp_directory_path() /
                (std::string("pqEmbeddedPython-") + info->name());
        std::filesystem::remove_all(_root);
        std::filesystem::create_directories(_root);
    }

    void TestEmbeddedPython::TearDown() { std::filesystem::remove_all(_root); }

    std::filesystem::path TestEmbeddedPython::_makeDir(const std::string &name
    ) const
    {
        auto path = _root / name;
        std::filesystem::create_directories(path);
        return path;
    }

    std::filesystem::path TestEmbeddedPython::_makeFile(
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
}   // namespace

TEST(TestPythonProgramName, matchesLinkedLibpython)
{
    EXPECT_EQ(
        utilities::pythonProgramName(),
        "python" + std::to_string(PY_MAJOR_VERSION) + "." +
            std::to_string(PY_MINOR_VERSION)
    );
}

TEST_F(TestEmbeddedPython, firstExecutableOnPathWins)
{
    const auto empty  = _makeDir("empty");
    const auto first  = _makeDir("first");
    const auto second = _makeDir("second");
    const auto build  = _makeFile(_root / "build-python", true);
    _makeFile(first / _programName, true);
    _makeFile(second / _programName, true);

    const auto searchPath =
        empty.string() + "::" + first.string() + ":" + second.string();

    EXPECT_EQ(
        utilities::findPythonProgram(searchPath, _programName, build),
        first / _programName
    );
}

TEST_F(TestEmbeddedPython, skipsFilesThatCannotRun)
{
    const auto noExec = _makeDir("noExec");
    const auto good   = _makeDir("good");
    _makeFile(noExec / _programName, false);
    _makeFile(good / _programName, true);
    std::filesystem::create_directories(_makeDir("isDir") / _programName);

    const auto searchPath = (_root / "isDir").string() + ":" +
                            noExec.string() + ":" + good.string();

    EXPECT_EQ(
        utilities::findPythonProgram(searchPath, _programName, {}),
        good / _programName
    );
}

TEST_F(TestEmbeddedPython, fallsBackToBuildPython)
{
    const auto empty = _makeDir("empty");
    const auto build = _makeFile(_root / "build-python", true);

    EXPECT_EQ(
        utilities::findPythonProgram(empty.string(), _programName, build),
        build
    );
    EXPECT_EQ(utilities::findPythonProgram("", _programName, build), build);
}

TEST_F(TestEmbeddedPython, nothingFoundLeavesPythonDefault)
{
    const auto empty = _makeDir("empty");

    EXPECT_FALSE(utilities::findPythonProgram(
                     empty.string(),
                     _programName,
                     _root / "missing-python"
    )
                     .has_value());
    EXPECT_FALSE(
        utilities::findPythonProgram(empty.string(), _programName, {})
            .has_value()
    );
}

#if !defined(_WIN32)
// death tests run in a child process, so each one starts its own interpreter
TEST(TestEmbeddedPythonDeathTest, startKeepsPQHandler)
{
    EXPECT_EXIT(
        {
            utilities::installInterruptHandler();
            utilities::startPython();
            utilities::startPython();
            const auto started = ::Py_IsInitialized() != 0;
            const auto handler = utilities::isInterruptHandlerInstalled();
            utilities::stopPython();
            utilities::stopPython();
            std::exit(
                started && handler && ::Py_IsInitialized() == 0 ? 0 : 1
            );
        },
        testing::ExitedWithCode(0),
        ""
    );
}

TEST(TestEmbeddedPythonDeathTest, startWithoutPQHandlerKeepsPythonHandler)
{
    EXPECT_EXIT(
        {
            static_cast<void>(std::signal(SIGINT, SIG_DFL));
            utilities::startPython();
            std::exit(utilities::isInterruptHandlerInstalled() ? 1 : 0);
        },
        testing::ExitedWithCode(0),
        ""
    );
}

TEST(TestEmbeddedPythonDeathTest, ctrlCBeforeStartRaisesKeyboardInterrupt)
{
    EXPECT_EXIT(
        {
            utilities::installInterruptHandler();
            static_cast<void>(std::raise(SIGINT));
            utilities::startPython();

            try
            {
                utilities::throwIfInterrupted();
            }
            catch (const pybind11::error_already_set &error)
            {
                std::exit(error.matches(PyExc_KeyboardInterrupt) ? 0 : 1);
            }

            std::exit(1);
        },
        testing::ExitedWithCode(0),
        ""
    );
}
#endif
