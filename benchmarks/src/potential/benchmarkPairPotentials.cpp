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

#include "buckinghamPair.hpp"
#include "coulombReactionField.hpp"
#include "coulombShiftedPotential.hpp"
#include "coulombWolf.hpp"
#include "lennardJonesPair.hpp"
#include "morsePair.hpp"

namespace
{
    template <typename PairPotential>
    void runNonCoulombBenchmark(
        benchmark::State& state,
        PairPotential&    potential
    )
    {
        double distance = static_cast<double>(state.range(0)) / 1000.0;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(distance);
            auto result = potential.calculate(distance);
            benchmark::DoNotOptimize(result);
        }
    }

    template <typename CoulombPotential>
    void runCoulombBenchmark(
        benchmark::State& state,
        CoulombPotential& potential
    )
    {
        double distance      = static_cast<double>(state.range(0)) / 1000.0;
        double chargeProduct = -0.25;

        for (auto _ : state)
        {
            benchmark::DoNotOptimize(distance);
            benchmark::DoNotOptimize(chargeProduct);
            auto result = potential.calculate(distance, chargeProduct);
            benchmark::DoNotOptimize(result);
        }
    }

    void BM_LennardJones(benchmark::State& state)
    {
        potential::LennardJonesPair potential(9.0, 2.0, 3.0);
        runNonCoulombBenchmark(state, potential);
    }

    void BM_Buckingham(benchmark::State& state)
    {
        potential::BuckinghamPair potential(9.0, 1.0, 0.3, 2.0);
        runNonCoulombBenchmark(state, potential);
    }

    void BM_Morse(benchmark::State& state)
    {
        potential::MorsePair potential(9.0, 1.0, 2.0, 1.5);
        runNonCoulombBenchmark(state, potential);
    }

    void BM_CoulombShifted(benchmark::State& state)
    {
        potential::CoulombShiftedPotential potential(9.0);
        runCoulombBenchmark(state, potential);
    }

    void BM_CoulombWolf(benchmark::State& state)
    {
        potential::CoulombWolf potential(9.0, 0.25);
        runCoulombBenchmark(state, potential);
    }

    void BM_CoulombReactionField(benchmark::State& state)
    {
        potential::CoulombReactionField potential(9.0, 78.5);
        runCoulombBenchmark(state, potential);
    }

    BENCHMARK(BM_LennardJones)
        ->ArgName("distance_milliangstrom")
        ->Arg(1500)
        ->Arg(3000)
        ->Arg(6000);
    BENCHMARK(BM_Buckingham)
        ->ArgName("distance_milliangstrom")
        ->Arg(1500)
        ->Arg(3000)
        ->Arg(6000);
    BENCHMARK(BM_Morse)
        ->ArgName("distance_milliangstrom")
        ->Arg(1500)
        ->Arg(3000)
        ->Arg(6000);
    BENCHMARK(BM_CoulombShifted)
        ->ArgName("distance_milliangstrom")
        ->Arg(1500)
        ->Arg(3000)
        ->Arg(6000);
    BENCHMARK(BM_CoulombWolf)
        ->ArgName("distance_milliangstrom")
        ->Arg(1500)
        ->Arg(3000)
        ->Arg(6000);
    BENCHMARK(BM_CoulombReactionField)
        ->ArgName("distance_milliangstrom")
        ->Arg(1500)
        ->Arg(3000)
        ->Arg(6000);
}   // namespace
