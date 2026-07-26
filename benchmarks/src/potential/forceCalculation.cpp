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
#include <memory>
#include <type_traits>

#include "benchmarkSetup.hpp"
#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "coulombShiftedPotential.hpp"
#include "guffNonCoulomb.hpp"
#include "lennardJonesPair.hpp"
#include "physicalData.hpp"
#include "potentialBruteForce.hpp"
#include "potentialCellList.hpp"
#include "potentialSettings.hpp"

namespace
{
    std::shared_ptr<potential::GuffNonCoulomb> makeNonCoulombPotential()
    {
        auto nonCoulomb = std::make_shared<potential::GuffNonCoulomb>();
        nonCoulomb->resizeGuff(1);
        nonCoulomb->resizeGuff(0, 1);
        nonCoulomb->resizeGuff(0, 0, 1);
        nonCoulomb->resizeGuff(0, 0, 0, 1);

        const auto pair = std::make_shared<potential::LennardJonesPair>(
            benchmarkSetup::cutOff,
            1.0,
            1.0
        );
        nonCoulomb->setGuffNonCoulPair({1, 1, 0, 0}, pair);

        return nonCoulomb;
    }

    template <typename PotentialType>
    void BM_ForceCalculation(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        settings::PotentialSettings::setCoulombRadiusCutOff(
            benchmarkSetup::cutOff
        );
        potential::CoulombPotential::setCoulombRadiusCutOff(
            benchmarkSetup::cutOff
        );
        potential::CoulombPotential::setCoulombEnergyCutOff(0.0);
        potential::CoulombPotential::setCoulombForceCutOff(0.0);

        PotentialType forceCalculation;
        forceCalculation.makeCoulombPotential(
            potential::CoulombShiftedPotential(benchmarkSetup::cutOff)
        );
        forceCalculation.setNonCoulombPotential(makeNonCoulombPotential());

        simulationBox::CellList cellList;
        if constexpr (std::is_same_v<
                          PotentialType,
                          potential::PotentialCellList>)
        {
            cellList.setNumberOfCells(cellsPerSide);
            cellList.resizeCells();
            cellList.setup(simBox);
            cellList.activate();
            cellList.updateCellList(simBox);
        }

        physicalData::PhysicalData physicalData;

        for (auto _ : state)
        {
            state.PauseTiming();
            benchmarkSetup::resetForces(simBox);
            state.ResumeTiming();

            forceCalculation.calculateForces(simBox, physicalData, cellList);
            benchmark::DoNotOptimize(physicalData.getCoulombEnergy());
            benchmark::DoNotOptimize(physicalData.getNonCoulombEnergy());
        }

        const auto numberOfMolecules =
            static_cast<std::int64_t>(simBox.getNumberOfMolecules());
        state.SetItemsProcessed(state.iterations() * numberOfMolecules);
        state.SetComplexityN(numberOfMolecules);
    }

    BENCHMARK_TEMPLATE(BM_ForceCalculation, potential::PotentialBruteForce)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(6)
        ->Arg(8)
        ->Arg(10)
        ->Complexity(benchmark::oNSquared);

    BENCHMARK_TEMPLATE(BM_ForceCalculation, potential::PotentialCellList)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12)
        ->Arg(16)
        ->Arg(24)
        ->Complexity(benchmark::oN);
}   // namespace
