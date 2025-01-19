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

#ifndef __MIN_MAX_SUM_MEAN_INL__
#define __MIN_MAX_SUM_MEAN_INL__

#include "max.inl"
#include "mean.inl"
#include "min.inl"
#include "sum.inl"

namespace linearAlgebra
{

    template <typename T>
    std::tuple<T, T, T, T> minMaxSumMean(
        T           *a,
        const size_t size,
        const bool   onDevice
    )
    {
        const auto _min  = min(a, size, onDevice);
        const auto _max  = max(a, size, onDevice);
        const auto _sum  = sum(a, size, onDevice);
        const auto _mean = mean(a, size, onDevice);

        return {_min, _max, _sum, _mean};
    }

    template <typename T>
    std::tuple<T, T, T, T> minMaxSumMean(
        const std::vector<T> &vector,
        const size_t          size,
        const bool            onDevice
    )
    {
        const auto _min  = min(vector, size, onDevice);
        const auto _max  = max(vector, size, onDevice);
        const auto _sum  = sum(vector, size, onDevice);
        const auto _mean = mean(vector, size, onDevice);

        return {_min, _max, _sum, _mean};
    }
}   // namespace linearAlgebra

#endif   // __MEAN_INL__