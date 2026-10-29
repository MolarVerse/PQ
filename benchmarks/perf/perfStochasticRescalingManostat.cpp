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

// Fixed-work micro-benchmark of the stochastic rescaling manostat, applied to
// 20 molecules of 3 atoms (pressure estimate, scaling matrix and scaling of box
// and molecules). Random numbers: fixed seed.

#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "perfBenchSetup.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "stochasticRescalingManostat.hpp"
#include "timingsSettings.hpp"

static constexpr std::uint64_t ITERATIONS = 1000*100*10;

int main()
{
    settings::TimingsSettings::setTimeStep(0.001);
    settings::PotentialSettings::setCoulombRadiusCutOff(4.0);
    settings::GeneralSettings::setRandomSeed(42);
    settings::GeneralSettings::setIsRandomSeedSet(true);

    auto box =
        benchSetup::makePopulatedBox({.nMolecules = 20, .nAtomsPerMol = 3});
    auto physicalData = physicalData::PhysicalData();
    auto manostat     = manostat::StochasticRescalingManostat(
        1.0,
        100.0,
        4.6e-5,
        FixedAxis::NONE
    );

    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        manostat.applyManostat(box, physicalData);
    }

    // read state so the loop cannot be optimized away
    std::cout << std::format("{:.6f}\n", box.getVolume());
    return 0;
}
