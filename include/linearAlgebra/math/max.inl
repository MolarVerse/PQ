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

#ifndef __MAX_INL__
#define __MAX_INL__

/**
 * @file max.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief API for the max function
 * @version 0.1
 * @date 2025-01-11
 *
 * @details This file contains the following functions:
 *      - maxHost
 *      - maxDevice
 *      - max
 */

#include <algorithm>   // for max_element
#include <cassert>
#include <cmath>
#include <vector>

#include "exceptions.hpp"

namespace linearAlgebra
{
    /**
     * @brief Calculates the maximum of all elements in an array
     *
     * @param a
     * @param size
     *
     * @return maximum of all elements in a
     */
    template <typename T>
    T maxHost(T* a, const size_t size)
    {
        return *std::max_element(a, a + size);
    }

    /**
     * @brief Calculates the maximum of all elements in an array on the device
     *
     * @param a
     * @param size
     *
     * @return maximum of all elements in a
     */
    template <typename T>
    T maxDevice(T* a, const size_t size)
    {
        T max = std::numeric_limits<T>::min();
        // clang-format off
        #pragma omp target teams distribute parallel for \
                    is_device_ptr(a)                     \
                    reduction(max:max)                   \
                    map(max)
        // clang-format on
        for (size_t i = 0; i < size; ++i) max = std::max(max, a[i]);

        return max;
    }

    /**
     * @brief Calculates the maximum of all elements in an array
     *
     * @param a
     * @param size
     * @param onDevice
     *
     * @return maximum of all elements in a
     */
    template <typename T>
    T max(T* a, const size_t size, const bool onDevice)
    {
        if (onDevice)
            return maxDevice(a, size);
        else
            return maxHost(a, size);
    }

    /**
     * @brief Calculates the maximum of all elements in a vector
     *
     * @param vector
     *
     * @return maximum of all elements in vector
     */
    template <typename T>
    T max(const std::vector<T>& vector)
    {
        return *std::ranges::max_element(vector);
    }

    /**
     * @brief Calculates the maximum of all elements in a vector
     *
     * @param a
     * @param size
     *
     * @return maximum of all elements in a
     */
    template <typename T>
    T max(const std::vector<T>& vector, const size_t size)
    {
        assert(vector.size() == size);
        return max(vector);
    }

    /**
     * @brief Calculates the maximum of all elements in a vector
     *
     * @param a
     * @param size
     * @param onDevice
     *
     * @return maximum of all elements in a
     */
    template <typename T>
    T max(const std::vector<T>& vector, const size_t size, const bool onDevice)
    {
        if (onDevice)
            throw customException::NotImplementedException(
                "max on device with vector"
            );
        else
            return max(vector, size);
    }

}   // namespace linearAlgebra

#endif   // __MAX_INL__