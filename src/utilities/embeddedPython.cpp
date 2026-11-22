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

#include "executablePath.hpp"

#if !defined(_WIN32)
#include <pthread.h>

#include <csignal>
#include <mstd/utils.hpp>

#include "interrupt.hpp"
#endif

#if !defined(_WIN32)
namespace
{
    /**
     * @brief hands SIGINT to Python while it starts and gives it back to PQ
     *
     * @details Python installs its handler only over SIG_DFL, so PQ's handler
     * is reset and SIGINT blocked until the guard ends
     */
    class SigintRelease
    {
       private:
        bool     _released = false;
        sigset_t _oldMask{};

       public:
        SigintRelease();
        ~SigintRelease();

        MSTD_NO_COPY(SigintRelease);
        MSTD_NO_MOVE(SigintRelease);
    };

    SigintRelease::SigintRelease()
    {
        if (!utilities::isInterruptHandlerInstalled())
        {
            return;
        }

        sigset_t sigintOnly{};
        ::sigemptyset(&sigintOnly);
        ::sigaddset(&sigintOnly, SIGINT);
        ::pthread_sigmask(SIG_BLOCK, &sigintOnly, &_oldMask);
        static_cast<void>(std::signal(SIGINT, SIG_DFL));
        _released = true;
    }

    SigintRelease::~SigintRelease()
    {
        if (_released)
        {
            utilities::installInterruptHandler();
            ::pthread_sigmask(SIG_SETMASK, &_oldMask, nullptr);
        }
    }
}   // namespace
#endif

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
        auto program = findInSearchPath(searchPath, programName);

        if (!program.has_value() && !buildPython.empty() &&
            isExecutableFile(buildPython))
        {
            program = buildPython;
        }

        return program;
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

#if !defined(_WIN32)
        const SigintRelease sigintRelease;
#endif
        pybind11::initialize_interpreter(&config);
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
