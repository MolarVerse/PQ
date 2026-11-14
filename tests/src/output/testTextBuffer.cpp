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

#include "textBuffer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <format>
#include <limits>
#include <string>

namespace
{
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    constexpr std::array values{
        0.0,     -0.0,       1.5,   -1.23456789,  123456789.123,
        1e-12,   5e-9,       -5e-9, 0.123456785,  2.5e-300,
        1e300,   -1.7e308,   inf,   -inf,         nan,
        -nan,    1.0 / 3.0,  42.0,  -99999.99999, 6.02214076e23,
    };
}   // namespace

TEST(TestTextBuffer, fixedMatchesFormat)
{
    out::TextBuffer buffer;
    std::string     expected;

    for (const double value : values)
    {
        buffer.fixed(value, 15, 8).put('\t');
        expected += std::format("{:15.8f}\t", value);
    }

    EXPECT_EQ(buffer.view(), expected);
}

TEST(TestTextBuffer, scientificMatchesFormat)
{
    out::TextBuffer buffer;
    std::string     expected;

    for (const double value : values)
    {
        buffer.scientific(value, 20, 8).scientific(value, 10, 5).put('\n');
        expected += std::format("{:20.8e}{:10.5e}\n", value, value);
    }

    EXPECT_EQ(buffer.view(), expected);
}

TEST(TestTextBuffer, integersAndTextMatchFormat)
{
    out::TextBuffer buffer;
    std::string     expected;

    for (const std::size_t value : {0UZ, 7UZ, 12345UZ, 1234567UZ})
    {
        buffer.left(value, 5).put('|');
        expected += std::format("{:<5}|", value);
    }

    for (const std::string name : {"", "O", "H1", "Na+  ", "LONGNAME"})
    {
        buffer.left(name, 5).append(name);
        expected += std::format("{:<5}{}", name, name);
    }

    EXPECT_EQ(buffer.view(), expected);
}

TEST(TestTextBuffer, clearKeepsNothing)
{
    out::TextBuffer buffer;

    buffer.fixed(1.0, 15, 8);
    buffer.clear();
    buffer.append("x");

    EXPECT_EQ(buffer.view(), "x");
}
