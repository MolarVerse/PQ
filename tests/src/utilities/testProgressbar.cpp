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

#include <sstream>
#include <string>

#include "progressbar.hpp"
#include "terminalProgressBar.hpp"

TEST(TestProgressbar, singleIterationCompletes)
{
    auto output = std::ostringstream();
    auto bar    = progressbar(1, true, output);

    bar.update();

    EXPECT_NE(output.str().find("100%"), std::string::npos);
}

TEST(TestProgressbar, terminalBarWritesProgress)
{
    auto output = std::ostringstream();
    auto bar    = utilities::TerminalProgressBar(1, output, true);

    bar.update();

    EXPECT_NE(output.str().find("100%"), std::string::npos);
}

TEST(TestProgressbar, redirectedOutputStaysEmpty)
{
    auto output = std::ostringstream();
    auto bar    = utilities::TerminalProgressBar(5000, output, false);

    for (int step = 0; step < 5000; ++step)
    {
        bar.update();
    }

    EXPECT_TRUE(output.str().empty());
}
