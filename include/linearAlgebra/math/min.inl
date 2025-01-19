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

#ifndef __MIN_INL__
#define __MIN_INL__

/**
 * @file min.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief API for the min function
 * @version 0.1
 * @date 2025-01-11
 *
 * @details This file contains the following functions:
 *      - minHost
 *      - minDevice
 *      - min
 *
 */

#include <algorithm>   // for min_element
#include <cassert>
#include <cmath>
#include <vector>   // for vector

#include "exceptions.hpp"

namespace linearAlgebra
{
    /**
     * @brief Calculates the minimum of all elements in an array
     *
     * @param a
     * @param size
     *
     * @return minimum of all elements in a
     */
    template <typename T>
    T minHost(T* a, const size_t size)
    {
        return *std::min_element(a, a + size);
    }

    /**
     * @brief Calculates the minimum of all elements in an array on the device
     *
     * @param a
     * @param size
     *
     * @return minimum of all elements in a
     */
    template <typename T>
    T minDevice(T* a, const size_t size)
    {
        T min = std::numeric_limits<T>::max();
        // clang-format off
        #pragma omp target teams distribute parallel for \
                    is_device_ptr(a)                     \
                    reduction(min:min)                   \
                    map(min)
        // clang-format on
        for (size_t i = 0; i < size; ++i) min = std::min(min, a[i]);

        return min;
    }

    /**
     * @brief Calculates the minimum of all elements in an array
     *
     * @param a
     * @param size
     * @param onDevice
     *
     * @return minimum of all elements in a
     */
    template <typename T>
    T min(T* a, const size_t size, const bool onDevice)
    {
        if (onDevice)
            return minDevice(a, size);
        else
            return minHost(a, size);
    }

    /**
     * @brief Calculates the minimum of all elements in an array
     *
     * @param a
     * @param size
     *
     * @return minimum of all elements in a
     */
    template <typename T>
    T min(T* a, const size_t size)
    {
        return min(a, size, false);
    }

    /**
     * @brief Calculates the minimum of all elements in a vector
     *
     * @param vector
     *
     * @return minimum of all elements in vector
     */
    template <typename T>
    T min(const std::vector<T>& vector)
    {
        return *std::ranges::min_element(vector);
    }

    /**
     * @brief Calculates the minimum of all elements in a vector
     *
     * @param vector
     * @param size
     *
     * @return minimum of all elements in vector
     */
    template <typename T>
    T min(const std::vector<T>& vector, const size_t size)
    {
        assert(vector.size() == size);
        return min(vector);
    }

    /**
     * @brief Calculates the minimum of all elements in a vector
     *
     * @param vector
     * @param size
     * @param onDevice
     *
     * @return minimum of all elements in vector
     */
    template <typename T>
    T min(const std::vector<T>& vector, const size_t size, const bool onDevice)
    {
        if (onDevice)
            throw customException::NotImplementedException(
                "min on device with vector"
            );
        else
            return min(vector, size);
    }

}   // namespace linearAlgebra

#endif   // __MIN_INL__
