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
#endif
