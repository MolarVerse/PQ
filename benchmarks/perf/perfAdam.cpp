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

// Fixed-work benchmark of the Adam update step on a box of atoms (moments,
// bias correction, position update and wrapping into the box).

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "adam.hpp"
#include "perfAlloc.hpp"
#include "perfOptimizers.hpp"

static constexpr std::uint64_t ITERATIONS = 200;
static constexpr std::size_t   ATOMS      = 4096;
static constexpr std::size_t   PER_SIDE   = 16;

int main()
{
    const auto box = benchOptimizers::makeBox(ATOMS, PER_SIDE);

    opt::Adam optimizer(ITERATIONS, ATOMS);
    optimizer.setSimulationBox(box);

    perfAllocBegin();
    CALLGRIND_ZERO_STATS;

    // the bias correction divides by (1 - beta^step), so the first step is 1
    for (std::uint64_t i = 1; i <= ITERATIONS; ++i) optimizer.update(0.001, i);

    std::cout << std::format("{:.6f}\n", benchOptimizers::sumOfPositions(*box));
    return 0;
}
