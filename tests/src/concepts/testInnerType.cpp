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

#include <optional>
#include <type_traits>
#include <vector>

#include "innerType.hpp"

namespace
{
    /**
     * @brief non-template, non-arithmetic type used to check that
     * InnerType_t/is_InnerType_arithmetic_v do not misclassify it
     *
     */
    struct NonArithmetic
    {
    };
}   // namespace

/**
 * @brief tests that InnerType_t resolves plain types to themselves
 *
 */
TEST(TestInnerType, plainType)
{
    EXPECT_TRUE((std::is_same_v<pq::InnerType_t<int>, int>) );
    EXPECT_TRUE((std::is_same_v<pq::InnerType_t<double>, double>) );
}

/**
 * @brief tests that InnerType_t unwraps a single template class
 *
 */
TEST(TestInnerType, singleTemplateClass)
{
    EXPECT_TRUE((std::is_same_v<pq::InnerType_t<std::optional<int>>, int>) );
    EXPECT_TRUE((std::is_same_v<pq::InnerType_t<std::vector<double>>, double>
    ) );
}

/**
 * @brief tests that InnerType_t recursively unwraps nested template classes
 *
 */
TEST(TestInnerType, nestedTemplateClass)
{
    EXPECT_TRUE((
        std::is_same_v<pq::InnerType_t<std::vector<std::optional<int>>>, int>
    ) );
}

/**
 * @brief tests is_InnerType_arithmetic_v
 *
 */
TEST(TestInnerType, isInnerTypeArithmeticV)
{
    EXPECT_TRUE(pq::is_InnerType_arithmetic_v<int>);
    EXPECT_TRUE(pq::is_InnerType_arithmetic_v<std::optional<double>>);
    EXPECT_FALSE(pq::is_InnerType_arithmetic_v<std::optional<NonArithmetic>>);
}

/**
 * @brief tests is_InnerType_quantity_like_v for plain (non-quantity) types
 *
 */
TEST(TestInnerType, isInnerTypeQuantityLikeV)
{
    EXPECT_FALSE(pq::is_InnerType_quantity_like_v<int>);
    EXPECT_FALSE(pq::is_InnerType_quantity_like_v<std::optional<double>>);
}
