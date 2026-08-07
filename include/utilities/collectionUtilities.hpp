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

#ifndef _COLLECTION_UTILITIES_HPP_

#define _COLLECTION_UTILITIES_HPP_

#include <algorithm>   // for ranges::sort, ranges::unique
#include <vector>      // for vector

namespace utilities
{
    /**
     * @brief returns sorted unique elements from a vector copy
     *
     * @tparam T
     * @param elements
     * @return std::vector<T>
     */
    template <typename T>
    [[nodiscard]] std::vector<T> getUniqueElements(std::vector<T> elements)
    {
        std::ranges::sort(elements);
        const auto [first, last] = std::ranges::unique(elements);

        elements.erase(first, last);

        return elements;
    }

}   // namespace utilities

#endif   // _COLLECTION_UTILITIES_HPP_
