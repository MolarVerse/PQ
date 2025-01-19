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

#ifndef _STL_VECTOR_HPP_

#define _STL_VECTOR_HPP_

#include <cmath>     // for sqrt
#include <numeric>   // for inner_product
#include <vector>    // for vector

namespace std
{
    /**
     * @brief dot product of two vectors
     *
     * @param a std::vector<T>
     * @param b std::vector<T>
     *
     * @return T dot product of a and b
     */
    template <typename T>
    T dot(const std::vector<T> &a, const std::vector<T> &b)
    requires std::is_arithmetic_v<T>
    {
        return std::inner_product(a.begin(), a.end(), b.begin(), T());
    }

    /**
     * @brief root mean square of a vector
     *
     * @param a std::vector<T>
     *
     * @return T root mean square of a
     */
    template <typename T>
    T rms(const std::vector<T> &a)
    {
        return std::sqrt(dot(a, a) / a.size());
    }

}   // namespace std

#endif   // _STL_VECTOR_HPP_