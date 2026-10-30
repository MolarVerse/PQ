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

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#endif

#include "box.hpp"
#include "constants/conversionFactors.hpp"
#include "exceptions.hpp"
#include "executablePath.hpp"
#include "fileSettings.hpp"
#include "generalSettings.hpp"
#include "globalTimer.hpp"
#include "physicalData.hpp"
#include "qmSettings.hpp"
#include "simulationBox.hpp"

#if !defined(_WIN32)
namespace
{
    struct CancelState
    {
        static inline volatile std::sig_atomic_t flag = 0;
    };
}   // namespace

/**
 * @brief signal handler that records SIGINT, SIGTERM or SIGHUP while an
 * external QM program runs
 *
 * @details signal handlers need C language linkage, which is ignored for names
 * with internal linkage, so this cannot live in the unnamed namespace
 *
 * @param signalNumber received signal
 */
// NOLINTNEXTLINE(misc-use-internal-linkage): C linkage requires external
// linkage
extern "C" void pqRecordQMCancelSignal(const int signalNumber)
{
    CancelState::flag = signalNumber;
}

namespace
{
    /**
     * @brief records SIGINT, SIGTERM and SIGHUP while the QM process runs, so
     * its process group can be stopped before PQ exits; restores the previous
     * handlers on destruction
     */
    class ScopedCancelSignals
    {
       private:
        static constexpr auto _nSignals = 3;

        std::array<int, _nSignals> _signals{SIGINT, SIGTERM, SIGHUP};
        std::array<struct sigaction, _nSignals> _previous{};

       public:
        ScopedCancelSignals()
        {
            CancelState::flag = 0;

            struct sigaction action{};
            action.sa_handler = pqRecordQMCancelSignal;
            ::sigemptyset(&action.sa_mask);

            for (std::size_t i = 0; i < _signals.size(); ++i)
            {
                ::sigaction(_signals.at(i), &action, &_previous.at(i));

                // keep signals ignored by the caller (e.g. nohup) ignored
                if (_previous.at(i).sa_handler == SIG_IGN)
                {
                    ::sigaction(_signals.at(i), &_previous.at(i), nullptr);
                }
            }
        }

        ~ScopedCancelSignals()
        {
            for (std::size_t i = 0; i < _signals.size(); ++i)
            {
                ::sigaction(_signals.at(i), &_previous.at(i), nullptr);
            }
        }

        ScopedCancelSignals(const ScopedCancelSignals &)            = delete;
        ScopedCancelSignals &operator=(const ScopedCancelSignals &) = delete;
        ScopedCancelSignals(ScopedCancelSignals &&)                 = delete;
        ScopedCancelSignals &operator=(ScopedCancelSignals &&)      = delete;
    };

    /**
     * @brief terminates the process group of pid and reaps pid
     *
     * @details sends SIGTERM and waits until pid exits, at most
     * terminateGracePeriod; SIGKILL then stops any remaining group members.
     * pid is reaped last so its process group id cannot be reused before.
     *
     * @param pid process (group leader) to stop
     * @param status wait status of pid
     * @param pollInterval interval between checks whether pid exited
     */
    void stopProcessGroup(
        const pid_t                     pid,
        int                            &status,
        const std::chrono::milliseconds pollInterval
    )
    {
        constexpr auto terminateGracePeriod = std::chrono::seconds(1);

        const auto deadline =
            std::chrono::steady_clock::now() + terminateGracePeriod;

        ::killpg(pid, SIGTERM);

        siginfo_t info{};
        while (std::chrono::steady_clock::now() < deadline)
        {
            info.si_pid = 0;
            ::waitid(
                P_PID,
                static_cast<id_t>(pid),
                &info,
                WEXITED | WNOHANG | WNOWAIT
            );
            if (info.si_pid == pid)
            {
                break;
            }

            std::this_thread::sleep_for(pollInterval);
        }

        ::killpg(pid, SIGKILL);
        ::waitpid(pid, &status, 0);
    }
}   // namespace
#endif

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
        molsys::Periodicity         per,
        const HybridSettings       &hybridSettings
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

        _setPeriodicity(per);

        {
            auto _ = scopedTimer(TimerId::QMEngine, "Write Coordinates");
            writeCoordsFile(simulationBox);
        }

        if (settings::GeneralSettings::isHybridJobtype())
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

        {
            auto _ =
                scopedTimer(TimerId::QMEngine, "Execute External QM Runner");
            execute(simulationBox, hybridSettings);
        }

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
        constexpr auto execFailedStatus = 127;
        constexpr auto pollInterval     = std::chrono::milliseconds(10);

        auto       shell         = std::string("sh");
        auto       shellFlag     = std::string("-c");
        auto       commandString = std::string(command);
        const auto timeLimit     = settings::QMSettings::getQMLoopTimeLimit();

        const pid_t pid = ::fork();
        if (pid < 0)
            throw exc::QMRunnerException(
                std::format("{} command could not be started", program)
            );

        if (pid == 0)
        {
            ::setpgid(0, 0);
            auto argv = std::array<char *, 4>{
                shell.data(),
                shellFlag.data(),
                commandString.data(),
                nullptr
            };
            ::execv("/bin/sh", argv.data());
            ::_exit(execFailedStatus);
        }

        ::setpgid(pid, pid);

        // qm_loop_time_limit is given in s
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::duration<double>(timeLimit);

        int status = 0;
        {
            const auto cancelSignals = ScopedCancelSignals{};

            while (true)
            {
                const auto result = ::waitpid(pid, &status, WNOHANG);

                if (result == pid)
                    break;

                if (result < 0 && errno != EINTR)
                    throw exc::QMRunnerException(
                        std::format("{} command could not be awaited", program)
                    );

                if (CancelState::flag != 0)
                {
                    stopProcessGroup(pid, status, pollInterval);
                    break;
                }

                if (timeLimit > 0.0 &&
                    std::chrono::steady_clock::now() >= deadline)
                {
                    stopProcessGroup(pid, status, pollInterval);
                    throw exc::QMRunnerException("QM calculation timeout");
                }

                std::this_thread::sleep_for(pollInterval);
            }
        }

        // previous handlers are restored; deliver the signal to PQ itself
        if (CancelState::flag != 0)
        {
            static_cast<void>(::raise(CancelState::flag));
        }

        if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS)
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
