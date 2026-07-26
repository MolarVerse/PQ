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

#include "velocityVerlet.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "benchmarkSetup.hpp"
#include "timingsSettings.hpp"

namespace
{
    void BM_VelocityVerletFirstStep(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        settings::TimingsSettings::setTimeStep(0.001);
        integrator::VelocityVerlet integrator;

        for (auto _ : state)
        {
            integrator.firstStep(simBox);
            benchmark::DoNotOptimize(simBox.getAtoms().data());
        }

        state.SetItemsProcessed(
            state.iterations() *
            static_cast<std::int64_t>(simBox.getNumberOfAtoms())
        );
    }

    void BM_VelocityVerletSecondStep(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        settings::TimingsSettings::setTimeStep(0.001);
        integrator::VelocityVerlet integrator;

        for (auto _ : state)
        {
            integrator.secondStep(simBox);
            benchmark::DoNotOptimize(simBox.getAtoms().data());
        }

        state.SetItemsProcessed(
            state.iterations() *
            static_cast<std::int64_t>(simBox.getNumberOfAtoms())
        );
    }

    BENCHMARK(BM_VelocityVerletFirstStep)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12);
    BENCHMARK(BM_VelocityVerletSecondStep)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12);
}   // namespace
