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

// Fixed-work micro-benchmark of the GUFF pair kernel with all 22 coefficients
// non-zero and with a sparse set (most real GUFF files have many zero
// coefficients, which the kernel skips).

#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include <algorithm>
#include <array>

#include "defaults.hpp"
#include "guffPair.hpp"

static constexpr std::uint64_t ITERATIONS = 20000;

int main()
{
    std::array<double, defaults::NUM_GUFF_COEFFICIENTS> dense{};
    std::array<double, defaults::NUM_GUFF_COEFFICIENTS> sparse{};
    std::ranges::generate(dense, [value = 2.0]() mutable { return value++; });
    sparse[0] = 2.0;
    sparse[1] = 6.0;
    sparse[2] = 3.0;
    sparse[3] = 12.0;

    const auto denseGuff  = pot::GuffPair(9.0, 1.0, 2.0, dense);
    const auto sparseGuff = pot::GuffPair(9.0, 1.0, 2.0, sparse);

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        const double distance =
            1.0 + (static_cast<double>(i & 255U) * 0.03);   // within cutoff
        const auto [eDense, fDense]    = denseGuff.calculate(distance);
        const auto [eSparse, fSparse]  = sparseGuff.calculate(distance);
        sink                          += eDense + fDense + eSparse + fSparse;
    }

    std::cout << std::format("{:.6f}\n", sink);
    return 0;
}
