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
#include <cstddef>
#include <limits>

#include "ulmJson.hpp"

/**
 * @brief numbers read back by Python's json module as the same float
 *
 */
TEST(TestUlmJson, number)
{
    EXPECT_EQ(out::ulm::number(10.0), "10.0");
    EXPECT_EQ(out::ulm::number(-1.5), "-1.5");
    EXPECT_EQ(out::ulm::number(0.1), "0.1");
    EXPECT_EQ(out::ulm::number(1e-5), "1e-05");
    EXPECT_EQ(out::ulm::number(std::numeric_limits<double>::quiet_NaN()), "NaN");
    EXPECT_EQ(
        out::ulm::number(std::numeric_limits<double>::infinity()),
        "Infinity"
    );
    EXPECT_EQ(
        out::ulm::number(-std::numeric_limits<double>::infinity()),
        "-Infinity"
    );
}

/**
 * @brief array entry as in ase/io/ulm.py
 *
 */
TEST(TestUlmJson, ndarray)
{
    constexpr auto matrix = std::array<std::size_t, 2>{3, 3};
    constexpr auto vector = std::array<std::size_t, 1>{3};

    EXPECT_EQ(
        out::ulm::ndarray(matrix, "float64", 80),
        R"({"ndarray": [[3, 3], "float64", 80]})"
    );
    EXPECT_EQ(
        out::ulm::ndarray(vector, "int64", 56),
        R"({"ndarray": [[3], "int64", 56]})"
    );
}
