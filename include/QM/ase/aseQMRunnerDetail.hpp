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

#ifndef QM__ASE__ASE_QM_RUNNER_DETAIL_HPP
#define QM__ASE__ASE_QM_RUNNER_DETAIL_HPP

#include <pybind11/numpy.h>

#include <atomic>
#include <stop_token>

#if !defined(_WIN32)
#include <pthread.h>
#endif

#include "box.hpp"
#include "simulationBox.hpp"
#include "unit_types.hpp"

namespace QM::detail
{
#if !defined(_WIN32)
    using PythonThread = pthread_t;
#else
    using PythonThread = int;
#endif

    [[nodiscard]] pybind11::array asePositions(
        const molsys::SimulationBox &simulationBox
    );

    [[nodiscard]] pybind11::array_t<double> aseCell(
        const molsys::SimulationBox &simulationBox
    );

    [[nodiscard]] pybind11::array_t<bool> asePBC(
        molsys::Periodicity periodicity
    );

    [[nodiscard]] pybind11::array_t<int> aseAtomicNumbers(
        const molsys::SimulationBox &simulationBox
    );

    [[nodiscard]] PythonThread currentThread();

    void interruptAfterTimeout(
        const std::stop_token &stopToken,
        std::atomic_bool      &timedOut,
        PythonThread           pythonThread,
        const TimeMs          &qmLoopTimeLimit
    );
}   // namespace QM::detail

#endif   // QM__ASE__ASE_QM_RUNNER_DETAIL_HPP
