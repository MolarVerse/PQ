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

#include "embeddedPython.hpp"

#include <pybind11/embed.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "interrupt.hpp"

#if !defined(_WIN32)
#include <pthread.h>

#include <csignal>
#endif

namespace
{
#if defined(_WIN32)
    constexpr char _pathSeparator = ';';
#else
    constexpr char _pathSeparator = ':';
#endif

    /**
     * @brief true if path is a regular file that someone may execute
     *
     * @param path candidate file
     */
    bool isExecutableFile(const std::filesystem::path &path)
    {
        std::error_code error;
        const auto      status = std::filesystem::status(path, error);

        if (error || !std::filesystem::is_regular_file(status))
        {
            return false;
        }

        constexpr auto anyExec = std::filesystem::perms::owner_exec |
                                 std::filesystem::perms::group_exec |
                                 std::filesystem::perms::others_exec;

        return (status.permissions() & anyExec) != std::filesystem::perms::none;
    }
#if !defined(_WIN32)
    /**
     * @brief block SIGINT and reset it to SIG_DFL: Python installs its
     * handler only over SIG_DFL
     *
     * @return the previous signal mask
     */
    sigset_t releaseSigint()
    {
        sigset_t sigintOnly{};
        sigset_t oldMask{};
        ::sigemptyset(&sigintOnly);
        ::sigaddset(&sigintOnly, SIGINT);
        ::pthread_sigmask(SIG_BLOCK, &sigintOnly, &oldMask);
        static_cast<void>(std::signal(SIGINT, SIG_DFL));

        return oldMask;
    }

    /**
     * @brief install PQ's SIGINT handler again and restore the signal mask
     *
     * @param oldMask mask returned by releaseSigint
     */
    void reclaimSigint(const sigset_t &oldMask)
    {
        utilities::installInterruptHandler();
        ::pthread_sigmask(SIG_SETMASK, &oldMask, nullptr);
    }
#endif
}   // namespace

namespace utilities
{
    /**
     * @brief name of the interpreter matching the linked libpython, e.g.
     * python3.12
     */
    std::string pythonProgramName()
    {
        return "python" + std::to_string(PY_MAJOR_VERSION) + "." +
               std::to_string(PY_MINOR_VERSION);
    }

    /**
     * @brief interpreter Python takes its prefix from: programName on
     * searchPath, else the build interpreter
     *
     * @param searchPath value of PATH
     * @param programName e.g. python3.12
     * @param buildPython interpreter PQ was built with, may be empty
     * @return std::nullopt if neither exists
     */
    std::optional<std::filesystem::path> findPythonProgram(
        std::string_view             searchPath,
        std::string_view             programName,
        const std::filesystem::path &buildPython
    )
    {
        while (!searchPath.empty())
        {
            const auto end       = searchPath.find(_pathSeparator);
            const auto directory = searchPath.substr(0, end);

            if (!directory.empty())
            {
                auto candidate = std::filesystem::path(directory) / programName;
                if (isExecutableFile(candidate))
                {
                    return candidate;
                }
            }

            if (end == std::string_view::npos)
            {
                break;
            }
            searchPath.remove_prefix(end + 1);
        }

        if (!buildPython.empty() && isExecutableFile(buildPython))
        {
            return buildPython;
        }

        return std::nullopt;
    }

    /**
     * @brief start the embedded interpreter once, keeping PQ's SIGINT handler
     *
     * @throw std::runtime_error if Python cannot be configured or started
     */
    void startPython()
    {
        if (::Py_IsInitialized() != 0)
        {
            return;
        }

        PyConfig config;
        ::PyConfig_InitPythonConfig(&config);
        config.parse_argv = 0;

        const auto *const searchPath = std::getenv("PATH");
        const auto        program    = findPythonProgram(
            searchPath == nullptr ? "" : searchPath,
            pythonProgramName(),
            PQ_PYTHON_EXECUTABLE
        );

        if (program.has_value())
        {
            const auto status = ::PyConfig_SetBytesString(
                &config,
                &config.program_name,
                program->string().c_str()
            );

            if (PyStatus_Exception(status) != 0)
            {
                ::PyConfig_Clear(&config);
                throw std::runtime_error("Failed to configure Python");
            }
        }

#if defined(_WIN32)
        pybind11::initialize_interpreter(&config);
#else
        std::optional<sigset_t> oldMask;
        if (isInterruptHandlerInstalled())
        {
            oldMask = releaseSigint();
        }

        try
        {
            pybind11::initialize_interpreter(&config);
        }
        catch (...)
        {
            if (oldMask.has_value())
            {
                reclaimSigint(*oldMask);
            }
            throw;
        }

        if (oldMask.has_value())
        {
            reclaimSigint(*oldMask);
        }
#endif
    }

    /**
     * @brief finalize the interpreter if a job started it
     */
    void stopPython()
    {
        if (::Py_IsInitialized() != 0)
        {
            pybind11::finalize_interpreter();
        }
    }
}   // namespace utilities
