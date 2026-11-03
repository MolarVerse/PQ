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

#include <string>

#include "base.hpp"

/**
 * @brief tests the Integral concept
 *
 */
TEST(TestConcepts, integral)
{
    EXPECT_TRUE(pq::Integral<int>);
    EXPECT_TRUE(pq::Integral<size_t>);
    EXPECT_FALSE(pq::Integral<double>);
    EXPECT_FALSE(pq::Integral<std::string>);
}

/**
 * @brief tests the FloatingPoint concept
 *
 */
TEST(TestConcepts, floatingPoint)
{
    EXPECT_TRUE(pq::FloatingPoint<double>);
    EXPECT_TRUE(pq::FloatingPoint<float>);
    EXPECT_FALSE(pq::FloatingPoint<int>);
    EXPECT_FALSE(pq::FloatingPoint<std::string>);
}

/**
 * @brief tests the Arithmetic concept
 *
 */
TEST(TestConcepts, arithmetic)
{
    EXPECT_TRUE(pq::Arithmetic<int>);
    EXPECT_TRUE(pq::Arithmetic<double>);
    EXPECT_FALSE(pq::Arithmetic<std::string>);
}

/**
 * @brief tests the is_arithmetic_v constexpr variable
 *
 */
TEST(TestConcepts, isArithmeticV)
{
    EXPECT_TRUE(pq::is_arithmetic_v<int>);
    EXPECT_TRUE(pq::is_arithmetic_v<double>);
    EXPECT_FALSE(pq::is_arithmetic_v<std::string>);
}
