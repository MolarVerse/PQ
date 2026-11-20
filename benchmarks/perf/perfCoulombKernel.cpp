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

#include "units.hpp"

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "perfAlloc.hpp"
#include "settings.hpp"

static constexpr std::uint64_t ITERATIONS = 20000;

int main()
{
    Settings settings;
    settings.potentialSettings.setCoulombRadiusCutOffByUser(9.0 * angstrom);
    auto shifted = pot::CoulombShiftedPotential(settings.potentialSettings);
    auto wolf = pot::CoulombWolf(settings.potentialSettings, 0.25 / angstrom);

    perfAllocBegin();

    CALLGRIND_ZERO_STATS;

    double sink = 0.0;
    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        const double distance =
            1.0 + (static_cast<double>(i & 255U) * 0.03);   // within cutoff
        const auto chargeProduct =
            0.5 * -0.5 * elementary_charge * elementary_charge;

        const auto [eShift, fShift] =
            shifted.calculate(distance * angstrom, chargeProduct);
        const auto [eWolf, fWolf] =
            wolf.calculate(distance * angstrom, chargeProduct);

        sink += (eShift + eWolf).numerical_value_in(kcal_per_mol) +
                (fWolf + fShift).numerical_value_in(kcal_per_mol / angstrom);
    }

    std::cout << std::format("{:.6f}\n", sink);
    return 0;
}
