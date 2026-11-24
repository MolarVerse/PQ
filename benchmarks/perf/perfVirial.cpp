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

// Fixed-work micro-benchmark of the molecular virial computation.

#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#include "unit_types.hpp"
#include "virial.hpp"

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "perfAlloc.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"

static constexpr std::uint64_t ITERATIONS = 1000;

int main()
{
    auto box =
        benchSetup::makePopulatedBox({.nMolecules = 20, .nAtomsPerMol = 3});

    auto physicalData = physicalData::PhysicalData();

    perfAllocBegin();

    CALLGRIND_ZERO_STATS;

    VirialPerMol result{VirialPerMol::zero()};

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        result = virial::calculateVirial(box);
        physicalData.set_virial(result);
    }

    std::cout << std::format(
        "{:.6f}\n",
        (result[0][0] + result[1][1] + result[2][2])
            .numerical_value_in(kcal_per_mol)
    );
    return 0;
}
