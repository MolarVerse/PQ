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

#ifndef __MEAN_INL__
#define __MEAN_INL__

/**
 * @file mean.inl
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief API for the mean function
 * @version 0.1
 * @date 2025-01-11
 *
 * @details This file contains the following functions:
 *      - meanHost
 *      - meanDevice
 *      - mean
 *
 */

#include <cassert>
#include <cmath>
#include <vector>

#include "exceptions.hpp"
#include "sum.inl"   // for sumHost, sumDevice

namespace linearAlgebra
{
    /**
     * @brief Calculates the mean of all elements in an array
     *
     * @param a
     * @param size
     *
     * @return mean of all elements in a
     */
    template <typename T>
    T meanHost(T* a, const size_t size)
    {
        T sum = 0.0;

        // clang-format off
    #pragma omp parallel for reduction(+ : sum)
        // clang-format on
        for (size_t i = 0; i < size; ++i) sum += a[i];

        return sum / size;
    }

    /**
     * @brief Calculates the mean of all elements in an array on the device
     *
     * @param a
     * @param size
     *
     * @return mean of all elements in a
     */
    template <typename T>
    T meanDevice(T* a, const size_t size)
    {
        T sum = 0.0;

        // clang-format off
    #pragma omp target teams distribute parallel for \
                reduction(+ : sum)                   \
                is_device_ptr(a)                     \
                map(sum)
        // clang-format on
        for (size_t i = 0; i < size; ++i) sum += a[i];

        return sum / size;
    }

    /**
     * @brief Calculates the mean of all elements in an array
     *
     * @param a
     * @param size
     * @param onDevice
     *
     * @return mean of all elements in a
     */
    template <typename T>
    T mean(T* a, const size_t size, const bool onDevice)
    {
        if (onDevice)
            return meanDevice(a, size);
        else
            return meanHost(a, size);
    }

    /**
     * @brief Calculates the mean of all elements in an array
     *
     * @param a
     * @param size
     *
     * @return mean of all elements in a
     */
    template <typename T>
    T mean(T* a, const size_t size)
    {
        return meanHost(a, size);
    }

    /**
     * @brief Calculates the mean of all elements in a vector
     *
     * @param vector
     *
     * @return mean of all elements in vector
     */
    template <typename T>
    T mean(const std::vector<T>& vector)
    {
        return sum(vector) / vector.size();
    }

    /**
     * @brief Calculates the mean of all elements in a vector
     *
     * @param vector
     * @param size
     *
     * @return mean of all elements in vector
     */
    template <typename T>
    T mean(const std::vector<T>& vector, const size_t size)
    {
        assert(vector.size() == size);
        return mean(vector);
    }

    /**
     * @brief Calculates the mean of all elements in a vector
     *
     * @param vector
     * @param size
     * @param onDevice
     *
     * @return mean of all elements in vector
     */
    template <typename T>
    T mean(const std::vector<T>& vector, const size_t size, const bool onDevice)
    {
        if (onDevice)
            throw customException::NotImplementedException(
                "mean on device with vector"
            );
        else
            return mean(vector, size);
    }

}   // namespace linearAlgebra

#endif   // __MEAN_INL__
