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

// Fixed-work micro-benchmark of the Berendsen manostat, applied to 20 molecules
// of 3 atoms (pressure estimate, scaling matrix and scaling of box and
// molecules).

#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#include "referenceSettings.hpp"
#include "settings.hpp"
#include "units.hpp"

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "berendsenManostat.hpp"
#include "perfAlloc.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"

static constexpr std::uint64_t ITERATIONS = 1000;

int main()
{
    const auto timeStep = 0.001 * fs;

    auto box =
        benchSetup::makePopulatedBox({.nMolecules = 20, .nAtomsPerMol = 3});
    auto physicalData = physicalData::PhysicalData();

    Settings settings;
    settings.potentialSettings.setCoulombRadiusCutoffByUser(4.0 * angstrom);
    settings.manostatSettings.set_targetPressureByUser(1.0 * bar);
    settings.manostatSettings.set_tauByUser(100.0 * fs);
    settings.manostatSettings.set_compressibilityByUser(4.6e-5 * per_bar);
    ReferenceSettings referenceSettings;
    settings.manostatSettings.finalize(referenceSettings);
    auto manostat = manostat::BerendsenManostat(settings.manostatSettings);

    perfAllocBegin();

    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        manostat.applyManostat(
            box,
            physicalData,
            toTemperature(300.0),
            timeStep,
            settings.potentialSettings.get_coulombRadiusCutoff()
        );
    }

    // read state so the loop cannot be optimized away
    std::cout << std::format(
        "{:.6f}\n",
        box.getVolume().numerical_value_in(angstrom * angstrom * angstrom)
    );
    return 0;
}
