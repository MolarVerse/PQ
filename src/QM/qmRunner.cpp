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

#include <cmath>
#include <thread>

#include "box.hpp"
#include "exceptions.hpp"
#include "qmSettings.hpp"
#include "units.hpp"

namespace QM
{

    /**
     * @brief constructor for the QMRunner class
     *
     * @param qmSettings the QMSettings object containing the settings for the
     * QM runner
     */
    QMRunner::QMRunner(const QMSettings &qmSettings) : _qmSettings(qmSettings)
    {
    }

    /**
     * @brief function to throw an exception after a timeout
     *
     * @details This function is used to throw an exception after a timeout. The
     * timeout is set in the settings file. If the timeout is set to 0, the
     * function will return without throwing an exception.
     *
     * @param stopToken The stop token used to check for cancellation of the QM
     * calculation loop
     * @param qmLoopTimeLimit The maximum allowed time for the QM calculation
     * loop. If the loop exceeds this time, an exception will be thrown. A value
     * of 0 indicates no time limit.
     *
     * @throw QMRunnerException if the timeout is exceeded
     */
    void QMRunner::throwAfterTimeout(
        const std::stop_token &stopToken,
        const TimeMs          &qmLoopTimeLimit
    )
    {
        if (qmLoopTimeLimit <= TimeMs::zero())
            return;

        const auto sleepDuration = 1 * ms;
        for (auto i = TimeMs::zero(); i < qmLoopTimeLimit; i += sleepDuration)
        {
            if (stopToken.stop_requested())
                return;

            std::this_thread::sleep_for(
                std::chrono::milliseconds(sleepDuration.numerical_value_in(ms))
            );
        }

        throw exc::QMRunnerException("QM calculation timeout");
    }

    /**
     * @brief run the qm engine with default periodicity XYZ (3d)
     *
     * @param simulationBox the simulation box to apply periodic boundary
     * conditions
     * @param physicalData the physical data associated with the simulation
     * @param fileSettings the file settings containing input/output file paths
     * and related configurations for the QM calculation
     */
    void QMRunner::run(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const FileSettings         &fileSettings
    )
    {
        run(simulationBox, physicalData, molsys::Periodicity::XYZ, fileSettings
        );
    }

    /**
     * @brief get the periodicity of the QM runner
     *
     * @return the periodicity
     */
    molsys::Periodicity QMRunner::_getPeriodicity() const
    {
        return _periodicity;
    }

    /**
     * @brief set the periodicity of the QM runner
     *
     * @param per the periodicity to set
     */
    void QMRunner::_setPeriodicity(molsys::Periodicity per)
    {
        _periodicity = per;
    }

    /**
     * @brief returns the QMSettings object associated with the QM runner
     *
     * @return const QMSettings&
     */
    const QMSettings &QMRunner::_getQMSettings() const
    {
        return _qmSettings.get();
    }

}   // namespace QM
