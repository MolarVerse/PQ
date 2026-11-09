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
#include <filesystem>
#include <string>
#include <vector>

#include "benchmarkSetup.hpp"
#include "trajectoryOutput.hpp"
#include "vector3d.hpp"

namespace
{
    using WriteFn = void (out::TrajectoryOutput::*)(
        const molsys::SimulationBox &,
        std::size_t,
        bool
    );

    // TMPDIR selects the filesystem under test
    std::string outputFile(const std::string &name)
    {
        return (std::filesystem::temp_directory_path() / name).string();
    }

    void setItemsProcessed(
        benchmark::State            &state,
        const molsys::SimulationBox &simulationBox
    )
    {
        state.SetItemsProcessed(
            state.iterations() *
            static_cast<std::int64_t>(simulationBox.getNumberOfAtoms())
        );
    }

    void writeFrames(
        benchmark::State  &state,
        WriteFn            write,
        const std::string &name
    )
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        const auto simBox       = benchmarkSetup::makeLattice(cellsPerSide);
        const auto fileName     = outputFile(name);

        {
            out::TrajectoryOutput output{fileName};
            output.setFilename(fileName, true);

            std::size_t step = 0;
            for (auto _ : state) (output.*write)(simBox, ++step, false);
        }

        state.SetBytesProcessed(
            static_cast<std::int64_t>(std::filesystem::file_size(fileName))
        );
        setItemsProcessed(state, simBox);
        std::filesystem::remove(fileName);
    }

    void BM_WriteXyz(benchmark::State &state)
    {
        writeFrames(state, &out::TrajectoryOutput::writeXyz, "pq-bench.xyz");
    }

    void BM_WriteVelocities(benchmark::State &state)
    {
        writeFrames(
            state,
            &out::TrajectoryOutput::writeVelocities,
            "pq-bench.vel"
        );
    }

    void BM_WriteForces(benchmark::State &state)
    {
        writeFrames(
            state,
            &out::TrajectoryOutput::writeForces,
            "pq-bench.force"
        );
    }

    void BM_WriteCharges(benchmark::State &state)
    {
        writeFrames(
            state,
            &out::TrajectoryOutput::writeCharges,
            "pq-bench.chrg"
        );
    }

    // what a writer thread would leave on the MD thread
    void BM_CopyFrame(benchmark::State &state)
    {
        const auto cellsPerSide = static_cast<std::size_t>(state.range(0));
        const auto simBox       = benchmarkSetup::makeLattice(cellsPerSide);
        const auto nAtoms       = simBox.getNumberOfAtoms();

        std::vector<linalg::Vec3D> positions(nAtoms);
        std::vector<linalg::Vec3D> velocities(nAtoms);
        std::vector<linalg::Vec3D> forces(nAtoms);

        for (auto _ : state)
        {
            for (std::size_t i = 0; i < nAtoms; ++i)
            {
                const auto &atom = simBox.getAtoms()[i];
                positions[i]     = atom->getPosition();
                velocities[i]    = atom->getVelocity();
                forces[i]        = atom->getForce();
            }

            benchmark::DoNotOptimize(positions.data());
            benchmark::DoNotOptimize(velocities.data());
            benchmark::DoNotOptimize(forces.data());
            benchmark::ClobberMemory();
        }

        setItemsProcessed(state, simBox);
    }
}   // namespace

// 216, 1728 and 13824 atoms
BENCHMARK(BM_WriteXyz)->Arg(6)->Arg(12)->Arg(24);
BENCHMARK(BM_WriteVelocities)->Arg(6)->Arg(12)->Arg(24);
BENCHMARK(BM_WriteForces)->Arg(6)->Arg(12)->Arg(24);
BENCHMARK(BM_WriteCharges)->Arg(6)->Arg(12)->Arg(24);
BENCHMARK(BM_CopyFrame)->Arg(6)->Arg(12)->Arg(24);
