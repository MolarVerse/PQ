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
#include "celllist.hpp"
#include "potentialSettings.hpp"
#include "settings.hpp"

namespace
{
    void BM_CellListUpdate(benchmark::State& state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        auto       simBox       = benchmarkSetup::makeLattice(cellsPerSide);

        settings::PotentialSettings::setCoulombRadiusCutOff(
            benchmarkSetup::cutOff
        );

        molsys::CellList cellList;
        cellList.setNumberOfCells(cellsPerSide);
        cellList.resizeCells();
        cellList.setup(simBox);
        settings::Settings::activateCellList();

        for (auto _ : state)
        {
            cellList.updateCellList(simBox);
            benchmark::DoNotOptimize(cellList.getCells().data());
        }

        state.SetItemsProcessed(
            state.iterations() *
            static_cast<std::int64_t>(simBox.getNumberOfMolecules())
        );
    }

    BENCHMARK(BM_CellListUpdate)
        ->ArgName("cells_per_side")
        ->Arg(5)
        ->Arg(8)
        ->Arg(12)
        ->Arg(16)
        ->Arg(24);
}   // namespace
