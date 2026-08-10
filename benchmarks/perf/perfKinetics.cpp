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

// Fixed-work micro-benchmark of the per-step kinetic diagnostics
// (temperature, momentum, angular momentum, total force).

#include <cstdio>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "perfBenchSetup.hpp"

static constexpr long ITERATIONS = 1000;

int main()
{
    auto box =
        benchSetup::makePopulatedBox({.nMolecules = 20, .nAtomsPerMol = 3});

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (long i = 0; i < ITERATIONS; ++i)
    {
        sink += box.calculateTemperature();
        sink += box.calculateMomentum()[0];
        sink += box.calculateAngularMomentum({0.0, 0.0, 0.0})[0];
        sink += box.calculateTotalForce();
    }

    std::cout << std::format("{:.6f}\n", sink);
    return 0;
}
