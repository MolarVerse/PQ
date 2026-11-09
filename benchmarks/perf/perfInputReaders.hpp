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

#ifndef BENCHMARKS__PERF__PERF_INPUT_READERS_HPP
#define BENCHMARKS__PERF__PERF_INPUT_READERS_HPP

// Shared body of the input reader scaling benchmarks: the moldescriptor, the
// restart file and the topology file (bonds and angles) of a box of water
// molecules are generated in a temporary directory outside the measured part
// and read into a fresh engine in every iteration. The cost per iteration
// should grow linearly with the number of molecules.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "fileSettings.hpp"
#include "forceFieldSettings.hpp"
#include "mmmdEngine.hpp"
#include "moldescriptorReader.hpp"
#include "perfAlloc.hpp"
#include "restartFileReader.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "topologyReader.hpp"

namespace benchInputReaders
{
    inline void writeFiles(
        const std::filesystem::path &directory,
        const std::size_t            molecules
    )
    {
        std::ofstream(directory / "moldescriptor.dat")
            << "# Molecule Type 1\n"
               "TIP3P-Water    3   0.0\n"
               "O              0  -0.834\n"
               "H              1   0.417\n"
               "H              1   0.417\n";

        const auto length = 3.1 * static_cast<double>(molecules);

        std::ofstream restart(directory / "input.rst");
        restart << "Step 0\n"
                << std::format("Box   {0} {0} {0}  90 90 90\n", length);

        for (std::size_t mol = 0; mol < molecules; ++mol)
        {
            const auto base = 3.1 * static_cast<double>(mol);
            for (const auto &[name, shift] :
                 {std::pair{"O", 0.0},
                  std::pair{"H", 0.9572},
                  std::pair{"H", -0.24}})
            {
                restart << std::format(
                    "{}   1   1   {:.8f}   {:.8f}   {:.8f}   "
                    "{:.8e}   {:.8e}   {:.8e}   {:.8f}   {:.8f}   {:.8f}\n",
                    name,
                    base + shift,
                    1.5,
                    2.5,
                    1.0e12,
                    -2.0e12,
                    3.0e12,
                    -1.5,
                    2.5,
                    -0.5
                );
            }
        }

        std::ofstream topology(directory / "topology.top");
        topology << "BONDS\n";
        for (std::size_t mol = 0; mol < molecules; ++mol)
        {
            const auto first = (3 * mol) + 1;
            topology << std::format(
                "{} {} 0\n{} {} 0\n",
                first,
                first + 1,
                first,
                first + 2
            );
        }
        topology << "END\n\nANGLES\n";
        for (std::size_t mol = 0; mol < molecules; ++mol)
        {
            const auto first = (3 * mol) + 1;
            topology
                << std::format("{} {} {} 0\n", first + 1, first, first + 2);
        }
        topology << "END\n";
    }

    inline int run(const std::size_t molecules, const std::uint64_t iterations)
    {
        const auto directory =
            std::filesystem::temp_directory_path() /
            std::format("pq_perf_input_readers_{}", molecules);
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        writeFiles(directory, molecules);

        Settings settings;
        settings.fileSettings.setMolDescriptorFileNameByUser(
            (directory / "moldescriptor.dat").string()
        );
        FileSettings::setStartFileName((directory / "input.rst").string());
        settings.fileSettings.setTopologyFileNameByUser(
            (directory / "topology.top").string()
        );

        settings.forceFieldSettings.setTypeByUser(ForceFieldType::BONDED);

        std::size_t atoms = 0;

        perfAllocBegin();

        CALLGRIND_ZERO_STATS;

        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            engine::MMMDEngine engine(settings);

            input::molDescriptor::MoldescriptorReader moldescriptor(engine);
            moldescriptor.read();
            input::restartFile::readRestartFile(engine);
            input::topology::readTopologyFile(engine);

            atoms = engine.getSimulationBox().getNumberOfAtoms();
        }

        std::filesystem::remove_all(directory);

        // read state so the loop cannot be optimized away
        std::cout << atoms << "\n";
        return 0;
    }
}   // namespace benchInputReaders

#endif   // BENCHMARKS__PERF__PERF_INPUT_READERS_HPP
