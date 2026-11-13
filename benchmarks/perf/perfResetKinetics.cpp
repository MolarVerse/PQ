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

// Fixed-work micro-benchmark of ResetKinetics: temperature, momentum and
// angular momentum reset plus the force reset, every iteration, on 60 atoms.

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

#include "perfAlloc.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"
#include "resetKinetics.hpp"
#include "resetKineticsSettings.hpp"

static constexpr std::uint64_t ITERATIONS = 1000;

int main()
{
    Temperature temperature{toTemperature(300.0)};

    auto box =
        benchSetup::makePopulatedBox({.nMolecules = 20, .nAtomsPerMol = 3});
    benchSetup::scaleVelocitiesToTemperature(box, temperature);

    auto physicalData = physicalData::PhysicalData();
    physicalData.setTemperature(temperature);

    // step <= n is always true here, so every reset runs in every iteration
    ResetKineticsSettings settings;
    settings.setNScaleByUser(ITERATIONS);
    settings.setFScaleByUser(1);
    settings.setNResetByUser(ITERATIONS);
    settings.setFResetByUser(1);
    settings.setNResetAngularByUser(ITERATIONS);
    settings.setFResetAngularByUser(1);
    settings.setFResetForcesByUser(1);
    const auto resetKinetics = kinetics::ResetKinetics(settings);

    perfAllocBegin();

    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        resetKinetics.reset(i, physicalData, box, temperature);
        resetKinetics.resetForces(i, box);
    }

    // read state so the loop cannot be optimized away
    std::cout << std::format(
        "{:.6f} {:.6f}\n",
        box.calculateTemperature().quantity_from_zero().numerical_value_in(K),
        box.calculateMomentum()[0]
    );
    return 0;
}
