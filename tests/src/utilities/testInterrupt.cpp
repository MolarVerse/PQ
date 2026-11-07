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

#include <csignal>
#include <cstdlib>

#include "interrupt.hpp"

#ifdef WITH_PYBIND11
#include <pybind11/embed.h>
#endif

TEST(TestInterrupt, noInterpreterDoesNotThrow)
{
    EXPECT_NO_THROW(utilities::throwIfInterrupted());
}

#ifdef WITH_PYBIND11
TEST(TestInterrupt, pendingSigintThrowsKeyboardInterrupt)
{
    const pybind11::scoped_interpreter guard{};

    EXPECT_NO_THROW(utilities::throwIfInterrupted());

    ASSERT_EQ(std::raise(SIGINT), 0);

    try
    {
        utilities::throwIfInterrupted();
        FAIL() << "pending SIGINT was not reported";
    }
    catch (const pybind11::error_already_set &error)
    {
        EXPECT_TRUE(error.matches(PyExc_KeyboardInterrupt));
    }

    EXPECT_NO_THROW(utilities::throwIfInterrupted());
}

#if !defined(_WIN32)
// death tests run in a child process, so the handler stays out of the others
TEST(TestInterruptDeathTest, firstCtrlCRaisesKeyboardInterrupt)
{
    EXPECT_EXIT(
        {
            const pybind11::scoped_interpreter guard{};
            utilities::installInterruptHandler();
            static_cast<void>(std::raise(SIGINT));

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

TEST(TestInterruptDeathTest, secondCtrlCExitsAtOnce)
{
    EXPECT_EXIT(
        {
            utilities::installInterruptHandler();
            static_cast<void>(std::raise(SIGINT));
            static_cast<void>(std::raise(SIGINT));
            std::exit(0);
        },
        testing::ExitedWithCode(128 + SIGINT),
        ""
    );
}

TEST(TestInterruptDeathTest, wakeIsNoCtrlC)
{
    EXPECT_EXIT(
        {
            utilities::installInterruptHandler();
            utilities::wakePythonThread(::pthread_self());
            static_cast<void>(std::raise(SIGINT));
            std::exit(0);
        },
        testing::ExitedWithCode(0),
        ""
    );
}
#endif
#endif
