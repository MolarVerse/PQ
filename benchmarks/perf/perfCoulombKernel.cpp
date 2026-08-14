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

// Fixed-work micro-benchmark of the Coulomb pair kernels (shifted + Wolf).

#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"

static constexpr std::uint64_t ITERATIONS = 20000;

int main()
{
    auto shifted = potential::CoulombShiftedPotential(9.0);
    auto wolf    = potential::CoulombWolf(9.0, 0.25);

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        const double distance =
            1.0 + static_cast<double>(i & 255) * 0.03;   // within cutoff
        const double chargeProduct = 0.5 * -0.5;

        const auto [eShift, fShift] =
            shifted.calculate(distance, chargeProduct);
        const auto [eWolf, fWolf] = wolf.calculate(distance, chargeProduct);

        sink += eShift + fShift + eWolf + fWolf;
    }

    std::cout << std::format("{:.6f}\n", sink);
    return 0;
}
