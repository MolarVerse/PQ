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

#include "qmRunner.hpp"

#include <cmath>    // for ceil
#include <thread>   // for sleep_for

#include "defaults.hpp"   // for _DIMENSIONALITY_DEFAULT_
#include "exceptions.hpp"
#include "qmSettings.hpp"

using QM::QMRunner;
using namespace settings;
using namespace defaults;
using namespace customException;

/**
 * @brief function to throw an exception after a timeout
 *
 * @details This function is used to throw an exception after a timeout. The
 * timeout is set in the settings file. If the timeout is set to 0, the function
 * will return without throwing an exception.
 *
 * @param stopToken
 *
 * @throw QMRunnerException if the timeout is exceeded
 */
void QMRunner::throwAfterTimeout(const std::stop_token stopToken) const
{
    const auto qmLoopTimeLimit = QMSettings::getQMLoopTimeLimit();

    if (qmLoopTimeLimit <= 0)
        return;

    const auto timeout = int(::ceil(qmLoopTimeLimit));

    for (int i = 0; i < timeout * 1000; ++i)
    {
        if (stopToken.stop_requested())
            return;

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    throw QMRunnerException("QM calculation timeout");
}

/**
 * @brief run the qm engine with default periodicity XYZ (3d)
 *
 * @param simBox SimulationBox reference
 * @param physicalData PhysicalData reference
 */
void QMRunner::run(pq::SimBox &simBox, pq::PhysicalData &physicalData)
{
    run(simBox, physicalData, Periodicity::XYZ);
}