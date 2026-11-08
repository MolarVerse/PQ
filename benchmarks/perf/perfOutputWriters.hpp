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

#ifndef BENCHMARKS__PERF__PERF_OUTPUT_WRITERS_HPP
#define BENCHMARKS__PERF__PERF_OUTPUT_WRITERS_HPP

// Shared body of the output writer scaling benchmarks: every iteration writes
// one frame of a box of water molecules (xyz, velocities, forces, charges,
// restart file and one energy line) to files in a temporary directory. The
// formatting is CPU work that callgrind counts; the cost per iteration should
// grow linearly with the number of molecules.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "units.hpp"

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "atom.hpp"
#include "berendsenThermostat.hpp"
#include "energyOutput.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "perfAlloc.hpp"
#include "physicalData.hpp"
#include "rstFileOutput.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "trajectoryOutput.hpp"
#include "vector3d.hpp"

namespace benchOutputWriters
{
    inline constexpr double SPACING = 3.1;

    inline int run(const std::size_t molecules, const std::uint64_t iterations)
    {
        settings::GeneralSettings::setJobtype(JobType::MM_MD);

        const auto perSide = static_cast<std::size_t>(
            std::ceil(std::cbrt(static_cast<double>(molecules)))
        );
        const double boxLength = SPACING * static_cast<double>(perSide);

        molsys::SimulationBox   simBox;
        molsys::OrthorhombicBox orthoBox{{boxLength, boxLength, boxLength}};
        simBox.setBox(orthoBox, false);

        const auto names = std::vector<std::string>{"O", "H", "H"};
        const auto shift = std::vector<linalg::Vec3D>{
            {0.0, 0.0, 0.0},
            {0.9572, 0.0, 0.0},
            {-0.24, 0.927, 0.0}
        };

        for (std::size_t mol = 0; mol < molecules; ++mol)
        {
            const auto column = mol % perSide;
            const auto row    = (mol / perSide) % perSide;
            const auto layer  = mol / (perSide * perSide);

            const linalg::Vec3D origin{
                SPACING * static_cast<double>(column),
                SPACING * static_cast<double>(row),
                SPACING * static_cast<double>(layer)
            };

            molsys::Molecule molecule;
            molecule.setMoltype(MolType{1});

            for (std::size_t i = 0; i < 3; ++i)
            {
                const auto index  = static_cast<double>(i);
                const auto offset = static_cast<double>(mol % 7) * 0.001;
                auto       atom   = std::make_shared<molsys::Atom>();
                atom->setName(names.at(i));
                atom->setPosition(origin + shift.at(i));
                atom->setVelocity({0.01 + offset, -0.02, 0.005 * index});
                atom->setForce({1.5 - offset, 0.25 * index, -0.75});
                atom->setPartialCharge(i == 0 ? -0.82 : 0.41);
                molecule.addAtom(atom);
                simBox.addAtom(atom);
            }

            simBox.addMolecule(molecule);
        }

        const auto directory =
            std::filesystem::temp_directory_path() /
            std::format("pq_perf_output_writers_{}", molecules);
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);

        const auto path = [&directory](const std::string &name)
        { return (directory / name).string(); };

        out::TrajectoryOutput xyz{path("traj.xyz")};
        out::TrajectoryOutput vel{path("traj.vel")};
        out::TrajectoryOutput force{path("traj.force")};
        out::TrajectoryOutput charge{path("traj.chg")};
        out::RstFileOutput    restart{path("traj.rst")};
        out::EnergyOutput     energy{path("traj.en")};
        for (auto *file : {&xyz, &vel, &force, &charge})
            file->setFilename(file->getFilename(), true);
        restart.setFilename(restart.getFilename(), true);
        energy.setFilename(energy.getFilename(), true);

        thermostat::BerendsenThermostat thermostat(
            mstd::units::point<K>(300.0),
            0.1 * fs
        );
        auto physicalData = physicalData::PhysicalData();
        physicalData.setTemperature(toTemperature(300.0));
        physicalData.setPressure(1.0);
        physicalData.setKineticEnergy(12.5);
        physicalData.setCoulombEnergy(-34.5);
        physicalData.setNonCoulombEnergy(7.25);
        Settings settings;

        perfAllocBegin();

        CALLGRIND_ZERO_STATS;

        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            xyz.writeXyz(simBox, i, true);
            vel.writeVelocities(simBox, i, true);
            force.writeForces(simBox, i, true);
            charge.writeCharges(simBox, i, true);
            restart.write(simBox, thermostat, i);
            energy.write(i, physicalData, settings);
        }

        xyz.close();
        vel.close();
        force.close();
        charge.close();
        restart.close();
        energy.close();

        // read state so the loop cannot be optimized away: the number of
        // bytes written
        std::uintmax_t bytes = 0;
        for (const auto &entry : std::filesystem::directory_iterator(directory))
            bytes += entry.file_size();

        std::filesystem::remove_all(directory);
        std::cout << bytes << "\n";
        return 0;
    }
}   // namespace benchOutputWriters

#endif   // BENCHMARKS__PERF__PERF_OUTPUT_WRITERS_HPP
