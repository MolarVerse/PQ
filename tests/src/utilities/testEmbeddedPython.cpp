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
#include <string>

#include "embeddedPython.hpp"
#include "interrupt.hpp"
#include "tempDirTest.hpp"

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace
{
    constexpr auto _programName = "python3.99";

    using FindPythonTest = TempDirTest;
}   // namespace

TEST(PythonProgramNameTest, matchesLinkedLibpython)
{
    EXPECT_EQ(
        utilities::pythonProgramName(),
        "python" + std::to_string(PY_MAJOR_VERSION) + "." +
            std::to_string(PY_MINOR_VERSION)
    );
}

TEST_F(FindPythonTest, searchPathWinsOverBuildPython)
{
    const auto bin   = _makeDir("bin");
    const auto build = _makeFile(_root / "build-python", true);
    _makeFile(bin / _programName, true);

    EXPECT_EQ(
        utilities::findPythonProgram(bin.string(), _programName, build),
        bin / _programName
    );
}

TEST_F(FindPythonTest, fallsBackToBuildPython)
{
    const auto noExec = _makeDir("noExec");
    const auto build  = _makeFile(_root / "build-python", true);
    _makeFile(noExec / _programName, false);

    EXPECT_EQ(
        utilities::findPythonProgram(noExec.string(), _programName, build),
        build
    );
    EXPECT_EQ(utilities::findPythonProgram("", _programName, build), build);
}

TEST_F(FindPythonTest, nothingFoundLeavesPythonDefault)
{
    const auto empty = _makeDir("empty");

    EXPECT_FALSE(
        utilities::findPythonProgram(
            empty.string(),
            _programName,
            _makeFile(_root / "build-python", false)
        )
            .has_value()
    );
    EXPECT_FALSE(
        utilities::findPythonProgram(
            empty.string(),
            _programName,
            _root / "missing-python"
        )
            .has_value()
    );
    EXPECT_FALSE(
        utilities::findPythonProgram(empty.string(), _programName, {})
            .has_value()
    );
}

#if !defined(_WIN32)
namespace
{
    using EmbeddedPythonDeathTest = TempDirTest;

    /**
     * @brief sys.executable of the running interpreter
     */
    std::string pythonExecutable()
    {
        return pybind11::module_::import("sys")
            .attr("executable")
            .cast<std::string>();
    }

    /**
     * @brief true if the calling thread blocks SIGINT
     */
    bool sigintBlocked()
    {
        sigset_t mask{};
        ::pthread_sigmask(SIG_SETMASK, nullptr, &mask);
        return ::sigismember(&mask, SIGINT) == 1;
    }
}   // namespace

// death tests run in a child process, so each one starts its own interpreter
TEST_F(EmbeddedPythonDeathTest, startKeepsPQHandler)
{
    EXPECT_EXIT(
        {
            utilities::installInterruptHandler();
            utilities::startPython();
            utilities::startPython();
            const auto started = ::Py_IsInitialized() != 0;
            const auto handler = utilities::isInterruptHandlerInstalled();
            const auto blocked = sigintBlocked();
            utilities::stopPython();
            utilities::stopPython();
            const auto stopped = ::Py_IsInitialized() == 0;
            std::exit(started && handler && !blocked && stopped ? 0 : 1);
        },
        testing::ExitedWithCode(0),
        ""
    );
}

TEST_F(EmbeddedPythonDeathTest, startWithoutPQHandlerKeepsPythonHandler)
{
    EXPECT_EXIT(
        {
            static_cast<void>(std::signal(SIGINT, SIG_DFL));
            utilities::startPython();
            const auto handler = utilities::isInterruptHandlerInstalled();
            std::exit(!handler && !sigintBlocked() ? 0 : 1);
        },
        testing::ExitedWithCode(0),
        ""
    );
}

TEST_F(EmbeddedPythonDeathTest, ctrlCBeforeStartRaisesKeyboardInterrupt)
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

TEST_F(EmbeddedPythonDeathTest, interpreterFromSearchPath)
{
    const auto bin     = _makeDir("bin");
    const auto program = bin / utilities::pythonProgramName();
    std::filesystem::create_symlink(PQ_PYTHON_EXECUTABLE, program);

    EXPECT_EXIT(
        {
            ::setenv("PATH", bin.c_str(), 1);
            utilities::startPython();
            std::exit(pythonExecutable() == program.string() ? 0 : 1);
        },
        testing::ExitedWithCode(0),
        ""
    );
}

TEST_F(EmbeddedPythonDeathTest, buildInterpreterWithoutSearchPath)
{
    const auto empty = _makeDir("empty");

    EXPECT_EXIT(
        {
            ::setenv("PATH", empty.c_str(), 1);
            utilities::startPython();
            std::exit(pythonExecutable() == PQ_PYTHON_EXECUTABLE ? 0 : 1);
        },
        testing::ExitedWithCode(0),
        ""
    );
}
#endif
