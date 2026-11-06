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

#ifndef BENCHMARKS__PERF__PERF_QM_RUNNER_HPP
#define BENCHMARKS__PERF__PERF_QM_RUNNER_HPP

// Shared body of the QM runner benchmarks: the file handling of one QM step of
// an external runner (DFTB+, Turbomole, PySCF) without starting a QM program.
// run() writes the coordinate file and the point charge file, builds the
// command, and reads the forces, charges and stress tensor. Instead of
// launching the QM program the command is replaced by copying canned result
// files into place.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "atom.hpp"
#include "externalQMRunner.hpp"
#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "perfAlloc.hpp"
#include "physicalData.hpp"
#include "qmSettings.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

namespace benchQMRunner
{
    inline constexpr std::size_t QM_MOLECULES = 16;
    inline constexpr std::size_t MM_MOLECULES = 400;
    inline constexpr std::size_t PER_SIDE     = 8;
    inline constexpr double      SPACING      = 3.1;

    // the runner without a QM program: copies the canned results into place
    template <typename Runner>
    class CannedResultsRunner : public Runner
    {
       private:
        std::filesystem::path _directory;

       public:
        explicit CannedResultsRunner(
            const QMSettings     &qmSettings,
            std::filesystem::path directory
        )
            : Runner(qmSettings), _directory(std::move(directory))
        {
        }

       protected:
        void _executeCommand(
            const std::string_view /*command*/,
            const std::string_view /*program*/
        ) const override
        {
            constexpr auto overwrite =
                std::filesystem::copy_options::overwrite_existing;
            std::filesystem::copy_file(
                _directory / "forces.canned",
                settings::FileSettings::getQMForcesTempFileName(),
                overwrite
            );
            std::filesystem::copy_file(
                _directory / "charges.canned",
                settings::FileSettings::getQMChargesTempFileName(),
                overwrite
            );
            std::filesystem::copy_file(
                _directory / "stress.canned",
                settings::FileSettings::getStressTensorTempFileName(),
                overwrite
            );
        }
    };

    inline void writeCannedResults(const std::filesystem::path &directory)
    {
        const auto atoms = 3 * QM_MOLECULES;

        std::ofstream forces(directory / "forces.canned");
        forces << "-76.123456789\n";
        for (std::size_t i = 0; i < atoms; ++i)
        {
            const auto value = 0.001 * static_cast<double>(i % 17);
            forces << std::format(
                "{:.9f} {:.9f} {:.9f}\n",
                value,
                -value,
                0.5 * value
            );
        }

        std::ofstream charges(directory / "charges.canned");
        for (std::size_t i = 0; i < atoms; ++i)
            charges << std::format("{:.9f}\n", (i % 3 == 0) ? -0.8 : 0.4);

        std::ofstream(directory / "stress.canned")
            << "1.0 0.1 0.2\n0.1 2.0 0.3\n0.2 0.3 3.0\n";
    }

    inline void buildBox(molsys::SimulationBox &simulationBox)
    {
        const auto length = SPACING * static_cast<double>(PER_SIDE);
        molsys::OrthorhombicBox orthoBox{{length, length, length}};
        simulationBox.setBox(orthoBox, false);

        const auto names  = std::vector<std::string>{"O", "H", "H"};
        const auto offset = std::vector<linalg::Vec3D>{
            {0.0, 0.0, 0.0},
            {0.9572, 0.0, 0.0},
            {-0.24, 0.927, 0.0}
        };

        for (std::size_t mol = 0; mol < QM_MOLECULES + MM_MOLECULES; ++mol)
        {
            const auto column = mol % PER_SIDE;
            const auto row    = (mol / PER_SIDE) % PER_SIDE;
            const auto layer  = mol / (PER_SIDE * PER_SIDE);

            const linalg::Vec3D origin{
                SPACING * static_cast<double>(column),
                SPACING * static_cast<double>(row),
                SPACING * static_cast<double>(layer)
            };

            molsys::Molecule molecule;
            molecule.setMoltype(MolType{1});

            for (std::size_t i = 0; i < 3; ++i)
            {
                auto atom = std::make_shared<molsys::Atom>();
                atom->setName(names.at(i));
                atom->setPosition(origin + offset.at(i));
                atom->setPartialCharge((i == 0) ? -0.82 : 0.41);
                molecule.addAtom(atom);
                simulationBox.addAtom(atom);
            }

            // the QM region is active, everything else is a point charge
            if (mol >= QM_MOLECULES)
            {
                molecule.setHybridZone(molsys::HybridZone::POINT_CHARGE);
                molecule.deactivateMolecule();
            }
            else
            {
                molecule.setHybridZone(molsys::HybridZone::CORE);
            }

            simulationBox.addMolecule(molecule);
        }
    }

    template <typename Runner>
    inline int run(
        const QMSettings         &qmSettings,
        const molsys::Periodicity periodicity,
        const std::uint64_t       iterations
    )
    {
        const auto directory =
            std::filesystem::temp_directory_path() /
            std::format(
                "pq_perf_qm_runner_{}",
                QMMethodMeta::toString(qmSettings.getQMMethod())
            );
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        writeCannedResults(directory);
        std::ofstream(directory / "script.sh") << "";

        const auto original = std::filesystem::current_path();
        std::filesystem::current_path(directory);

        settings::GeneralSettings::setJobtype(JobType::QMMM_MD);
        QMSettings::setRemoveNetForce(true);
        QMSettings::setQMLoopTimeLimit(0.0);
        QMSettings::setQMScript("script.sh");

        molsys::SimulationBox simulationBox;
        buildBox(simulationBox);

        CannedResultsRunner<Runner> runner(qmSettings, directory);
        runner.setScriptPath(directory.string() + "/");
        auto physicalData = physicalData::PhysicalData();

        perfAllocBegin();

        CALLGRIND_ZERO_STATS;

        for (std::uint64_t i = 0; i < iterations; ++i)
            runner.run(simulationBox, physicalData, periodicity);

        std::filesystem::current_path(original);
        std::filesystem::remove_all(directory);

        // read state so the loop cannot be optimized away
        std::cout << std::format("{:.6f}\n", physicalData.getQMEnergy());
        return 0;
    }
}   // namespace benchQMRunner

#endif   // BENCHMARKS__PERF__PERF_QM_RUNNER_HPP
