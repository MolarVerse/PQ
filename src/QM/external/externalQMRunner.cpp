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

#include "externalQMRunner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <thread>

#include "box.hpp"
#include "constants/conversionFactors.hpp"
#include "exceptions.hpp"
#include "executablePath.hpp"
#include "fileSettings.hpp"
#include "globalTimer.hpp"
#include "physicalData.hpp"
#include "qmSettings.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"

namespace QM
{

    /**
     * @brief reads the force file (including qm energy) and sets the forces of
     * the atoms
     *
     * @param simulationBox Simulation box containing molecules and atoms.
     * @param physicalData
     *
     * @throw QMRunnerException
     *  - if the force file cannot be opened
     *  - if the force file is empty
     */
    void ExternalQMRunner::_readForceFile(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData
    )
    {
        const auto forceFileName =
            settings::FileSettings::getQMForcesTempFileName();

        std::ifstream forceFile(forceFileName);

        if (!forceFile.is_open())
        {
            throw exc::QMRunnerException(
                std::format(
                    "Cannot open {} force file \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    forceFileName
                )
            );
        }

        if (forceFile.peek() == std::ifstream::traits_type::eof())
        {
            throw exc::QMRunnerException(
                std::format(
                    "Empty {} force file \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    forceFileName
                )
            );
        }

        double energy = 0.0;

        if (!(forceFile >> energy))
        {
            throw exc::QMRunnerException(
                std::format(
                    "Cannot read QM energy from {} force file \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    forceFileName
                )
            );
        }

        if (!std::isfinite(energy))
        {
            throw exc::QMRunnerException(
                std::format(
                    "Invalid QM energy (NaN/Inf) in {} force file \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    forceFileName
                )
            );
        }

        physicalData.setQMEnergy(energy * HARTREE_TO_KCAL_PER_MOL);

        auto readForces = [&forceFile, &forceFileName](auto &atom)
        {
            auto grad = linalg::Vec3D();

            if (!(forceFile >> grad[0] >> grad[1] >> grad[2]))
            {
                throw exc::QMRunnerException(
                    std::format(
                        "Incomplete {} force file \"{}\"",
                        QMMethodMeta::toString(
                            settings::QMSettings::getQMMethod()
                        ),
                        forceFileName
                    )
                );
            }

            for (size_t i = 0; i < 3; ++i)
            {
                if (!std::isfinite(grad[i]))
                {
                    throw exc::QMRunnerException(
                        std::format(
                            "Invalid QM force component (NaN/Inf) in {} force "
                            "file "
                            "\"{}\"",
                            QMMethodMeta::toString(
                                settings::QMSettings::getQMMethod()
                            ),
                            forceFileName
                        )
                    );
                }
            }

            atom->setForce(
                -grad * HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM
            );
        };

        std::ranges::for_each(simulationBox.getQMAtoms(), readForces);

        forceFile.close();

        if (settings::QMSettings::getRemoveNetForce())
            simulationBox.removeNetForce();
    }

    /**
     * @brief reads the charge file (qm_charges) and sets the _qmCharge of the
     * atoms
     *
     * @param simulationBox Simulation box containing molecules and atoms.
     *
     * @throw QMRunnerException
     *  - if the charge file cannot be opened
     *  - if the charge file is empty
     */
    void ExternalQMRunner::_readChargeFile(molsys::SimulationBox &simulationBox)
    {
        const auto chargeFileName =
            settings::FileSettings::getQMChargesTempFileName();

        std::ifstream chargeFile(chargeFileName);

        if (!chargeFile.is_open())
        {
            throw exc::QMRunnerException(
                std::format(
                    "Cannot open {} charge file \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    chargeFileName
                )
            );
        }

        if (chargeFile.peek() == std::ifstream::traits_type::eof())
        {
            throw exc::QMRunnerException(
                std::format(
                    "Empty {} charge file \"{}\"",
                    QMMethodMeta::toString(settings::QMSettings::getQMMethod()),
                    chargeFileName
                )
            );
        }

        simulationBox.resetQMCharges();

        auto readCharges = [&chargeFile, &chargeFileName](auto &atom)
        {
            auto charge = 0.0;

            if (!(chargeFile >> charge))
            {
                throw exc::QMRunnerException(
                    std::format(
                        "Incomplete {} charge file \"{}\"",
                        QMMethodMeta::toString(
                            settings::QMSettings::getQMMethod()
                        ),
                        chargeFileName
                    )
                );
            }
            if (!std::isfinite(charge))
            {
                throw exc::QMRunnerException(
                    std::format(
                        "Invalid value in {} charge file \"{}\"",
                        QMMethodMeta::toString(
                            settings::QMSettings::getQMMethod()
                        ),
                        chargeFileName
                    )
                );
            }

            atom->setQMCharge(charge);
        };

        std::ranges::for_each(simulationBox.getQMAtoms(), readCharges);

        chargeFile.close();
    }

    std::string bundledQMScriptPath(const std::string_view script)
    {
        const auto installedPath = utilities::installedDataPath(
            std::filesystem::path("scripts") / script
        );
        if (std::filesystem::is_regular_file(installedPath))
            return installedPath.string();

        return (std::filesystem::path(SCRIPT_PATH_) / script).string();
    }

    /**
     * @brief run the qm engine
     *
     * @param simulationBox SimulationBox reference
     * @param physicalData PhysicalData reference
     * @param per periodicity of the system
     */
    void ExternalQMRunner::run(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        molsys::Periodicity         per
    )
    {
        if (per != molsys::Periodicity::XYZ &&
            per != molsys::Periodicity::NON_PERIODIC)
        {
            throw exc::QMRunnerException(
                "External QM runners only available for non- and 3D-periodic "
                "calculations."
            );
        }

        _periodicity = per;

        {
            auto _ = scopedTimer(TimerId::QMEngine, "Write Coordinates");
            writeCoordsFile(simulationBox);
        }

        if (settings::Settings::isHybridJobtype())
        {
            auto _ = scopedTimer(TimerId::QMEngine, "Write Pointcharges");
            writePointChargeFile(simulationBox);
        }

        const auto resultFiles = std::array{
            settings::FileSettings::getQMForcesTempFileName(),
            settings::FileSettings::getQMChargesTempFileName(),
            settings::FileSettings::getStressTensorTempFileName()
        };
        for (const auto &file : resultFiles) std::filesystem::remove(file);

        std::jthread timeoutThread{[](const std::stop_token &stopToken)
                                   { throwAfterTimeout(stopToken); }};

        {
            auto _ =
                scopedTimer(TimerId::QMEngine, "Execute External QM Runner");
            execute(simulationBox);
        }

        timeoutThread.request_stop();

        {
            auto _ = scopedTimer(TimerId::QMEngine, "Read Forces");
            _readForceFile(simulationBox, physicalData);
        }

        {
            auto _ = scopedTimer(TimerId::QMEngine, "Read Charges");
            _readChargeFile(simulationBox);
        }

        if (per != molsys::Periodicity::NON_PERIODIC)
        {
            auto _ = scopedTimer(TimerId::QMEngine, "Read Stress Tensor");
            readStressTensor(simulationBox.getBox(), physicalData);
        }
    }

    std::string ExternalQMRunner::_resolveScriptPath(
        const std::string_view script
    ) const
    {
        if (_scriptPath.empty())
            return std::string(script);

        if (_scriptPath == SCRIPT_PATH_)
            return bundledQMScriptPath(script);

        return _scriptPath + std::string(script);
    }

    void ExternalQMRunner::_executeCommand(
        const std::string_view command,
        const std::string_view program
    ) const
    {
#if defined(_WIN32)
        static_cast<void>(command);
        throw QMRunnerException(
            std::format(
                "{} command execution is not supported on Windows",
                program
            )
        );
#else
        const auto status = std::system(std::string(command).c_str());
        if (status != EXIT_SUCCESS)
            throw exc::QMRunnerException(
                std::format("{} command failed with status {}", program, status)
            );
#endif
    }

    /********************************
     *                              *
     * standard getters and setters *
     *                              *
     ********************************/

    /**
     * @brief getter for the script path
     *
     * @return const std::string&
     */
    const std::string &ExternalQMRunner::getScriptPath() const
    {
        return _scriptPath;
    }

    /**
     * @brief getter for the singularity path
     *
     * @return  std::string
     */
    std::string ExternalQMRunner::getSingularity() { return _singularity; }

    /**
     * @brief getter for the static build path
     *
     * @return  std::string
     */
    std::string ExternalQMRunner::getStaticBuild() { return _staticBuild; }

    /**
     * @brief setter for the script path
     *
     * @param scriptPath
     */
    void ExternalQMRunner::setScriptPath(const std::string_view &scriptPath)
    {
        _scriptPath = scriptPath;
    }

}   // namespace QM
