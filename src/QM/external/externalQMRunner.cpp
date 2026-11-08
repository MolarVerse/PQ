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

namespace
{

    /**
     * @brief enumerates the types of files handled by the QM runner
     */
    enum class FileType : std::uint8_t
    {
        Force,
        Charge
    };

    /**
     * @brief throws a QMRunnerException for file errors
     *
     * @param message the error message
     * @param fileName the name of the file
     * @param qmMethod the QM method used
     * @param fileType the type of the file (force or charge)
     *
     * @throw QMRunnerException
     */
    void throwFileException(
        const std::string &message,
        const std::string &fileName,
        const std::string &qmMethod,
        FileType           fileType

    )
    {
        std::string fileTypeStr;
        switch (fileType)
        {
            case FileType::Force: fileTypeStr = "force"; break;
            case FileType::Charge: fileTypeStr = "charge"; break;
        }

        throw exc::QMRunnerException(
            std::format(
                "{} {} {} file \"{}\"",
                message,
                qmMethod,
                fileTypeStr,
                fileName
            )
        );
    }

    /**
     * @brief throws a QMRunnerException for force file errors
     *
     * @param message the error message
     * @param forceFileName the name of the force file
     * @param qmMethod the QM method used
     *
     * @throw QMRunnerException
     */
    void throwForceFileException(
        const std::string &message,
        const std::string &forceFileName,
        const std::string &qmMethod
    )
    {
        throwFileException(message, forceFileName, qmMethod, FileType::Force);
    }

    /**
     * @brief throws a QMRunnerException for charge file errors
     *
     * @param message the error message
     * @param chargeFileName the name of the charge file
     * @param qmMethod the QM method used
     *
     * @throw QMRunnerException
     */
    void throwChargeFileException(
        const std::string &message,
        const std::string &chargeFileName,
        const std::string &qmMethod
    )
    {
        throwFileException(message, chargeFileName, qmMethod, FileType::Charge);
    }

}   // namespace

namespace QM
{

    /**
     * @brief reads the force file (including qm energy) and sets the forces of
     * the atoms
     *
     * @param simulationBox Simulation box containing molecules and atoms.
     * @param physicalData The physical data object where the QM energy and
     * forces will be stored
     * @param qmSettings The QM settings used to configure the external QM
     * runner, which may include method-specific parameters and computational
     * options
     *
     * @throw QMRunnerException
     *  - if the force file cannot be opened
     *  - if the force file is empty
     */
    void ExternalQMRunner::_readForceFile(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const QMSettings           &qmSettings
    )
    {
        const auto forceFileName =
            settings::FileSettings::getQMForcesTempFileName();

        std::ifstream forceFile(forceFileName);

        if (!forceFile.is_open())
        {
            const auto &qmMethod = qmSettings.getQMMethod();
            throwForceFileException(
                "Cannot open",
                forceFileName,
                QMMethodMeta::toString(qmMethod)
            );
        }

        if (forceFile.peek() == std::ifstream::traits_type::eof())
        {
            throwForceFileException(
                "Empty",
                forceFileName,
                QMMethodMeta::toString(qmSettings.getQMMethod())
            );
        }

        double energy = 0.0;

        if (!(forceFile >> energy))
        {
            throwForceFileException(
                "Cannot read QM energy from",
                forceFileName,
                QMMethodMeta::toString(qmSettings.getQMMethod())
            );
        }

        if (!std::isfinite(energy))
        {
            throwForceFileException(
                "Invalid QM energy (NaN/Inf) in",
                forceFileName,
                QMMethodMeta::toString(qmSettings.getQMMethod())
            );
        }

        physicalData.setQMEnergy(energy * HARTREE_TO_KCAL_PER_MOL);

        auto readForces = [&forceFile, &forceFileName, &qmSettings](auto &atom)
        {
            auto grad = linalg::Vec3D();

            if (!(forceFile >> grad[0] >> grad[1] >> grad[2]))
            {
                throwForceFileException(
                    "Incomplete",
                    forceFileName,
                    QMMethodMeta::toString(qmSettings.getQMMethod())
                );
            }

            for (size_t i = 0; i < 3; ++i)
            {
                if (!std::isfinite(grad[i]))
                {
                    throwForceFileException(
                        "Invalid QM force component (NaN/Inf) in",
                        forceFileName,
                        QMMethodMeta::toString(qmSettings.getQMMethod())
                    );
                }
            }

            atom->setForce(
                -grad * HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM
            );
        };

        std::ranges::for_each(simulationBox.getQMAtoms(), readForces);

        forceFile.close();

        if (QMSettings::getRemoveNetForce())
            simulationBox.removeNetForce();
    }

    /**
     * @brief reads the charge file (qm_charges) and sets the _qmCharge of the
     * atoms
     *
     * @param simulationBox Simulation box containing molecules and atoms.
     * @param qmSettings The QM settings used to configure the external QM
     * runner, which may include method-specific parameters and computational
     * options
     *
     * @throw QMRunnerException
     *  - if the charge file cannot be opened
     *  - if the charge file is empty
     */
    void ExternalQMRunner::_readChargeFile(
        molsys::SimulationBox &simulationBox,
        const QMSettings      &qmSettings
    )
    {
        const auto chargeFileName =
            settings::FileSettings::getQMChargesTempFileName();

        std::ifstream chargeFile(chargeFileName);

        if (!chargeFile.is_open())
        {
            throwChargeFileException(
                "Cannot open",
                chargeFileName,
                QMMethodMeta::toString(qmSettings.getQMMethod())
            );
        }

        if (chargeFile.peek() == std::ifstream::traits_type::eof())
        {
            throwChargeFileException(
                "Empty",
                chargeFileName,
                QMMethodMeta::toString(qmSettings.getQMMethod())
            );
        }

        simulationBox.resetQMCharges();

        auto readCharges =
            [&chargeFile, &chargeFileName, &qmSettings](auto &atom)
        {
            auto charge = 0.0;

            if (!(chargeFile >> charge))
            {
                throwChargeFileException(
                    "Incomplete",
                    chargeFileName,
                    QMMethodMeta::toString(qmSettings.getQMMethod())
                );
            }
            if (!std::isfinite(charge))
            {
                throwChargeFileException(
                    "Invalid value in",
                    chargeFileName,
                    QMMethodMeta::toString(qmSettings.getQMMethod())
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
            execute(simulationBox);
        }

        {
            auto _ = scopedTimer(TimerId::QMEngine, "Read Forces");
            _readForceFile(simulationBox, physicalData, _getQMSettings());
        }

        {
            auto _ = scopedTimer(TimerId::QMEngine, "Read Charges");
            _readChargeFile(simulationBox, _getQMSettings());
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
        const auto timeLimit     = QMSettings::getQMLoopTimeLimit();

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
