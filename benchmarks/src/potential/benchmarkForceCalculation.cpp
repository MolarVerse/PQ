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
#include "generalSettings.hpp"
#include "guffNonCoulomb.hpp"
#include "lennardJonesPair.hpp"
#include "physicalData.hpp"
#include "potentialBruteForce.hpp"
#include "potentialCellList.hpp"
#include "settings.hpp"
#include "units.hpp"

namespace
{
    std::shared_ptr<pot::GuffNonCoulomb> makeNonCoulombPotential()
    {
        auto nonCoulomb = std::make_shared<pot::GuffNonCoulomb>();
        nonCoulomb->resizeGuff(1);
        nonCoulomb->resizeGuff(0, 1);
        nonCoulomb->resizeGuff(0, 0, 1);
        nonCoulomb->resizeGuff(0, 0, 0, 1);

        const auto pair = std::make_shared<pot::LennardJonesPair>(
            benchmarkSetup::cutOff,
            LJParams{
                .c6  = 1.0 * kcal_per_mol * angstrom6,
                .c12 = 1.0 * kcal_per_mol * angstrom12
            }
        );
        nonCoulomb->setGuffNonCoulPair(
            {MolType{1}, MolType{1}, AtomType{0}, AtomType{0}},
            pair
        );

        return nonCoulomb;
    }

    template <typename PotentialType>
    void BM_ForceCalculation(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        Settings settings;
        settings.potentialSettings.setCoulombRadiusCutoffByUser(
            benchmarkSetup::cutOff
        );

        PotentialType forceCalculation{settings.hybridSettings};
        forceCalculation.makeCoulombPotential(
            pot::CoulombShiftedPotential(settings.potentialSettings)
        );
        forceCalculation.setNonCoulombPotential(makeNonCoulombPotential());

        molsys::CellList cellList;
        if constexpr (std::is_same_v<PotentialType, pot::PotentialCellList>)
        {
            GeneralSettings::activateCellList();
            cellList.setNumberOfCells(cellsPerSide);
            cellList.resizeCells();
            cellList.setup(simBox, benchmarkSetup::cutOff);
            cellList.updateCellList(simBox, benchmarkSetup::cutOff);
        }

        physicalData::PhysicalData physicalData;

        for (auto _ : state)
        {
            state.PauseTiming();
            benchmarkSetup::resetForces(simBox);
            state.ResumeTiming();

            forceCalculation.calculateForces(
                simBox,
                physicalData,
                cellList,
                settings.waterModelSettings,
                settings.potentialSettings
            );
            auto tmp = physicalData.getCoulombEnergy();
            benchmark::DoNotOptimize(tmp);
            auto tmpNonCoulomb = physicalData.getNonCoulombEnergy();
            benchmark::DoNotOptimize(tmpNonCoulomb);
        }

        const auto numberOfMolecules =
            static_cast<std::int64_t>(simBox.getNumberOfMolecules());
        state.SetItemsProcessed(state.iterations() * numberOfMolecules);
        state.SetComplexityN(numberOfMolecules);
    }

    BENCHMARK_TEMPLATE(BM_ForceCalculation, pot::PotentialBruteForce)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(6)
        ->Arg(8)
        ->Arg(10)
        ->Complexity(benchmark::oNSquared);

    BENCHMARK_TEMPLATE(BM_ForceCalculation, pot::PotentialCellList)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12)
        ->Arg(16)
        ->Arg(24)
        ->Complexity(benchmark::oN);
}   // namespace
