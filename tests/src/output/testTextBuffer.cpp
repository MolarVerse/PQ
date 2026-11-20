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

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>

#include "textBuffer.hpp"

namespace
{
    constexpr auto inf = std::numeric_limits<double>::infinity();
    constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    constexpr std::array values{
        0.0,   -0.0,      1.5,   -1.23456789,  123456789.123,
        1e-12, 5e-9,      -5e-9, 0.123456785,  2.5e-300,
        1e300, -1.7e308,  inf,   -inf,         nan,
        -nan,  1.0 / 3.0, 42.0,  -99999.99999, 6.02214076e23,
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

    for (const std::string_view name : {"", "O", "H1", "Na+  ", "LONGNAME"})
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

namespace
{
    // the written text of a finite value reads back as the same double
    double readBack(const std::string_view text)
    {
        const auto first = text.find_first_not_of(' ');
        double     value = 0.0;

        const auto result = std::from_chars(
            text.data() + first,
            text.data() + text.size(),
            value
        );

        EXPECT_EQ(result.ec, std::errc()) << text;
        return value;
    }

    bool sameBits(const double lhs, const double rhs)
    {
        return std::bit_cast<std::uint64_t>(lhs) ==
               std::bit_cast<std::uint64_t>(rhs);
    }
}   // namespace

TEST(TestTextBuffer, exactFixedReadsBackAsTheSameDouble)
{
    for (const double value : values)
    {
        if (!std::isfinite(value))
            continue;

        out::TextBuffer buffer;
        buffer.exactFixed(value, 0);

        EXPECT_TRUE(sameBits(readBack(buffer.view()), value))
            << std::format("{}", value) << " -> " << buffer.view();
    }
}

TEST(TestTextBuffer, exactScientificReadsBackAsTheSameDouble)
{
    for (const double value : values)
    {
        if (!std::isfinite(value))
            continue;

        out::TextBuffer buffer;
        buffer.exactScientific(value, 0);

        EXPECT_TRUE(sameBits(readBack(buffer.view()), value))
            << std::format("{}", value) << " -> " << buffer.view();
    }
}

TEST(TestTextBuffer, exactTextIsTheShortestOne)
{
    out::TextBuffer fixed;
    fixed.exactFixed(0.1, 0).put(' ').exactFixed(1.0 / 3.0, 0).put(' ');
    fixed.exactFixed(42.0, 0);

    out::TextBuffer scientific;
    scientific.exactScientific(0.1, 0).put(' ');
    scientific.exactScientific(1.0 / 3.0, 0).put(' ');
    scientific.exactScientific(42.0, 0);

    // 0.1 is not written as 0.10000000000000001
    EXPECT_EQ(fixed.view(), "0.1 0.3333333333333333 42");
    EXPECT_EQ(scientific.view(), "1e-01 3.333333333333333e-01 4.2e+01");
}

TEST(TestTextBuffer, exactValuesAreRightAlignedAndNeverCut)
{
    out::TextBuffer buffer;

    buffer.exactFixed(1.5, 8).put('|');
    buffer.exactFixed(-10.987707231352603, 8).put('|');
    buffer.exactScientific(1.0, 9).put('|');
    buffer.exactScientific(5.1956552679409985e11, 9);

    // a field is padded up to the width and grows beyond it, no digit is lost
    EXPECT_EQ(
        buffer.view(),
        "     1.5|-10.987707231352603|    1e+00|5.1956552679409985e+11"
    );
}

TEST(TestTextBuffer, exactSpecialValuesLikeTheFixedFormat)
{
    for (const double value :
         {inf, -inf, std::numeric_limits<double>::quiet_NaN()})
    {
        out::TextBuffer exactBuffer;
        out::TextBuffer fixedBuffer;

        exactBuffer.exactFixed(value, 15);
        fixedBuffer.fixed(value, 15, 8);

        EXPECT_EQ(exactBuffer.view(), fixedBuffer.view());
    }
}
