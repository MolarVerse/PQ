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

#include <gtest/gtest.h>   // for Test, EXPECT_EQ

#include <string>   // for string
#include <vector>   // for vector

#include "collectionUtilities.hpp"   // for getUniqueElements

/**
 * @brief getUniqueElements sorts elements and removes duplicates.
 */
TEST(TestCollectionUtilities, getUniqueElements)
{
    const auto elements       = std::vector<size_t>{3u, 1u, 2u, 1u, 3u};
    const auto uniqueElements = utilities::getUniqueElements(elements);

    EXPECT_EQ(uniqueElements, (std::vector<size_t>{1u, 2u, 3u}));
}

/**
 * @brief getUniqueElements supports string vectors.
 */
TEST(TestCollectionUtilities, getUniqueElementsWithStrings)
{
    const auto elements = std::vector<std::string>{"O", "H", "O", "C", "H"};
    const auto uniqueElements = utilities::getUniqueElements(elements);

    EXPECT_EQ(uniqueElements, (std::vector<std::string>{"C", "H", "O"}));
}
