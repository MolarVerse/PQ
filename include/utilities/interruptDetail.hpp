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

#ifndef UTILITIES__INTERRUPT_DETAIL_HPP
#define UTILITIES__INTERRUPT_DETAIL_HPP

#if defined(WITH_PYBIND11) && !defined(_WIN32)
#include <csignal>

namespace utilities::detail
{
    /**
     * @brief Ctrl-C state shared by the SIGINT handler and wakePythonThread
     */
    struct InterruptState
    {
        static inline volatile std::sig_atomic_t interrupted = 0;
        static inline volatile std::sig_atomic_t wakeOnly    = 0;
    };
}   // namespace utilities::detail
#endif

#endif   // UTILITIES__INTERRUPT_DETAIL_HPP
