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

// Fixed-work benchmark of the optimizer bookkeeping of every step: copying
// forces and positions into the history, RMS and maximum force, and the
// convergence check.

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>

#include "units.hpp"

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "perfAlloc.hpp"
#include "perfOptimizers.hpp"
#include "physicalData.hpp"
#include "steepestDescent.hpp"

static constexpr std::uint64_t ITERATIONS = 200;
static constexpr std::size_t   ATOMS      = 4096;
static constexpr std::size_t   PER_SIDE   = 16;

int main()
{
    const auto box          = benchOptimizers::makeBox(ATOMS, PER_SIDE);
    auto       physicalData = physicalData::PhysicalData{};

    opt::SteepestDescent optimizer;

    // two entries are needed before the convergence check
    optimizer.updateHistory(physicalData.getTotalEnergy(), box);

    perfAllocBegin();
    CALLGRIND_ZERO_STATS;

    auto converged = 0U;
    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        physicalData.set_kineticEnergy(
            1.0 / static_cast<double>(i + 1) * kcal_per_mol
        );
        optimizer.updateHistory(physicalData.getTotalEnergy(), box);
        converged += optimizer.hasConverged() ? 1U : 0U;
    }

    std::cout << std::format(
        "{:.6f} {}\n",
        optimizer.getRMSForce().numerical_value_in(kcal_per_mol / angstrom),
        converged
    );
    return 0;
}
