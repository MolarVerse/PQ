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

// Fixed-work micro-benchmark of the stochastic velocity rescaling thermostat,
// applied to 20 molecules of 3 atoms. Random numbers: fixed seed.

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

#include "generalSettings.hpp"
#include "perfAlloc.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"
#include "velocityRescalingThermostat.hpp"

static constexpr std::uint64_t ITERATIONS = 1000;

int main()
{
    const auto      timeStep = 0.001 * fs;
    GeneralSettings generalSettings;
    generalSettings.setRandomSeedByUser(42);

    auto box =
        benchSetup::makePopulatedBox({.nMolecules = 20, .nAtomsPerMol = 3});
    auto       physicalData      = physicalData::PhysicalData();
    const auto targetTemperature = toTemperature(300.0);
    benchSetup::scaleVelocitiesToTemperature(box, targetTemperature);
    auto thermostat =
        thermostat::VelocityRescalingThermostat(100.0 * fs, generalSettings);

    perfAllocBegin();

    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        thermostat
            .applyThermostat(box, physicalData, targetTemperature, timeStep);
    }

    // read state so the loop cannot be optimized away
    std::cout << std::format(
        "{:.6f} {:.6f}\n",
        physicalData.getTemperature().quantity_from_zero().numerical_value_in(
            K
        ),
        box.calculateMomentum()[0].numerical_value_in(g_per_mol * angstrom / fs)
    );
    return 0;
}
