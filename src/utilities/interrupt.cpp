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

#include "interrupt.hpp"

#ifdef WITH_PYBIND11
#include <pybind11/pybind11.h>
#endif

#if defined(WITH_PYBIND11) && !defined(_WIN32)
#include <unistd.h>

#include <csignal>

namespace
{
    volatile std::sig_atomic_t _interrupted = 0;
    volatile std::sig_atomic_t _wakeOnly    = 0;

    /**
     * @brief the first Ctrl-C stops PQ at the next step, the second one
     * exits at once
     */
    extern "C" void onInterrupt(int)
    {
        if (_wakeOnly != 0)
        {
            _wakeOnly = 0;
            return;
        }

        if (_interrupted != 0)
            ::_exit(128 + SIGINT);

        _interrupted = 1;

        if (::Py_IsInitialized() != 0)
            ::PyErr_SetInterrupt();
    }
}   // namespace
#endif

namespace utilities
{
    /**
     * @brief throw if a signal recorded by embedded Python is pending
     *
     * @throw pybind11::error_already_set e.g. KeyboardInterrupt
     */
    void throwIfInterrupted()
    {
#ifdef WITH_PYBIND11
        if (::Py_IsInitialized() != 0 && ::PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set();
#endif
    }

#if defined(WITH_PYBIND11) && !defined(_WIN32)
    /**
     * @brief replace Python's SIGINT handler, which cannot stop PQ while
     * Python waits in native code
     */
    void installInterruptHandler()
    {
        struct sigaction action{};
        action.sa_handler = onInterrupt;
        ::sigemptyset(&action.sa_mask);
        ::sigaction(SIGINT, &action, nullptr);
    }

    /**
     * @brief wake thread from blocking calls such as time.sleep, without
     * counting as Ctrl-C
     */
    void wakePythonThread(pthread_t thread)
    {
        struct sigaction current{};
        ::sigaction(SIGINT, nullptr, &current);

        if (current.sa_handler == SIG_DFL || current.sa_handler == SIG_IGN)
            return;

        if (current.sa_handler == onInterrupt)
            _wakeOnly = 1;

        ::pthread_kill(thread, SIGINT);
    }
#endif
}   // namespace utilities
