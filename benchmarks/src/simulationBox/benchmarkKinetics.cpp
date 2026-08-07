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

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "benchmarkSetup.hpp"
#include "vector3d.hpp"

namespace
{
    void setItemsProcessed(
        benchmark::State&                   state,
        const simulationBox::SimulationBox& simBox
    )
    {
        state.SetItemsProcessed(
            state.iterations() *
            static_cast<std::int64_t>(simBox.getNumberOfAtoms())
        );
    }

    void BM_Temperature(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        for (auto _ : state)
        {
            auto result = simBox.calculateTemperature();
            benchmark::DoNotOptimize(result);
        }

        setItemsProcessed(state, simBox);
    }

    void BM_Momentum(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        for (auto _ : state)
        {
            auto result = simBox.calculateMomentum();
            benchmark::DoNotOptimize(result);
        }

        setItemsProcessed(state, simBox);
    }

    void BM_AngularMomentum(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);
        linearAlgebra::Vec3D momentum{0.1, -0.2, 0.3};

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(momentum);
            auto result = simBox.calculateAngularMomentum(momentum);
            benchmark::DoNotOptimize(result);
        }

        setItemsProcessed(state, simBox);
    }

    void BM_TotalForce(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        for (auto _ : state)
        {
            auto result = simBox.calculateTotalForce();
            benchmark::DoNotOptimize(result);
        }

        setItemsProcessed(state, simBox);
    }

    BENCHMARK(BM_Temperature)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12);
    BENCHMARK(BM_Momentum)->ArgName("cells_per_side")->Arg(5)->Arg(8)->Arg(12);
    BENCHMARK(BM_AngularMomentum)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12);
    BENCHMARK(BM_TotalForce)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12);
}   // namespace
