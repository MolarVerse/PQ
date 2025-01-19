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

#ifndef __SUM_INL__
#define __SUM_INL__

/**
 * @file sum.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief API for the sum function
 * @version 0.1
 * @date 2025-01-11
 *
 * @details This file contains the following functions:
 *      - sumHost
 *      - sumDevice
 *      - sum
 */

#include <cassert>
#include <cstddef>   // for size_t
#include <numeric>   // for std::accumulate
#include <vector>    // for std::vector

#include "exceptions.hpp"

namespace linearAlgebra
{
    /**
     * @brief Calculates the sum of all elements in an array
     *
     * @param a
     * @param size
     *
     * @return sum of all elements in a
     */
    template <typename T>
    T sumHost(T *a, const size_t size)
    {
        T sum = 0.0;

        // clang-format off
    #pragma omp parallel for reduction(+ : sum)
        // clang-format on
        for (size_t i = 0; i < size; ++i) sum += a[i];

        return sum;
    }

    /**
     * @brief Calculates the sum of all elements in an array on the device
     *
     * @param a
     * @param size
     *
     * @return sum of all elements in a
     */
    template <typename T>
    T sumDevice(T *a, const size_t size)
    {
        T sum = 0.0;

        // clang-format off
    #pragma omp target teams distribute parallel for   \
                reduction(+ : sum)                     \
                is_device_ptr(a)                       \
                map(sum)
        // clang-format on
        for (size_t i = 0; i < size; ++i) sum += a[i];

        return sum;
    }

    /**
     * @brief Calculates the sum of all elements in an array
     *
     * @param a
     * @param size
     * @param onDevice
     *
     * @return sum of all elements in a
     */
    template <typename T>
    T sum(T *a, const size_t size, const bool onDevice)
    {
        if (onDevice)
            return sumDevice(a, size);
        else
            return sumHost(a, size);
    }

    /**
     * @brief Calculates the sum of all elements in an array
     *
     * @param a
     * @param size
     *
     * @return sum of all elements in a
     */
    template <typename T>
    T sum(T *a, const size_t size)
    {
        return sum(a, size, false);
    }

    /**
     * @brief Calculates the sum of all elements in a vector
     *
     * @param vector
     *
     * @return sum of all elements in vector
     */
    template <typename T>
    T sum(const std::vector<T> &vector)
    {
        return std::accumulate(vector.begin(), vector.end(), T());
    }

    /**
     * @brief Calculates the sum of all elements in a vector
     *
     * @param a
     * @param size
     *
     * @return sum of all elements in a
     */
    template <typename T>
    T sum(const std::vector<T> &vector, const size_t size)
    {
        assert(vector.size() == size);
        return sum(vector);
    }

    /**
     * @brief Calculates the sum of all elements in a vector
     *
     * @param a
     * @param size
     * @param onDevice
     *
     * @return sum of all elements in a
     */
    template <typename T>
    T sum(const std::vector<T> &vector, const size_t size, const bool onDevice)
    {
        if (onDevice)
            throw customException::NotImplementedException(
                "sum on device with vector"
            );
        else
            return sum(vector, size);
    }

}   // namespace linearAlgebra

#endif   // __SUM_INL__