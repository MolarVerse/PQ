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

#include "engine.hpp"

#include "constants/conversionFactors.hpp"   // for _FS_TO_PS_
#include "logOutput.hpp"                     // for LogOutput
#include "outputFileSettings.hpp"            // for OutputFileSettings
#include "progressbar.hpp"                   // for progressbar
#include "referencesOutput.hpp"              // for ReferencesOutput
#include "stdoutOutput.hpp"                  // for StdoutOutput
#include "timingsSettings.hpp"               // for TimingsSettings
#include "vector3d.hpp"                      // for norm

using namespace engine;

/**
 * @brief Run the simulation for numberOfSteps steps.
 *
 */
void Engine::run()
{
    _timings.beginTimer();

    _physicalData.calculateKinetics(getSimulationBox());

    _engineOutput.getLogOutput().writeInitialMomentum(norm(_physicalData.getMomentum()));

    const auto  numberOfSteps = settings::TimingsSettings::getNumberOfSteps();
    progressbar bar(static_cast<int>(numberOfSteps), true, std::cout);

    for (; _step <= numberOfSteps; ++_step)
    {
        bar.update();
        takeStep();

        writeOutput();
    }

    _timings.endTimer();

    const auto elapsedTime = _timings.calculateElapsedTime() * 1e-3;

    references::ReferencesOutput::writeReferencesFile();

    _engineOutput.getLogOutput().writeEndedNormally(elapsedTime);
    _engineOutput.getStdoutOutput().writeEndedNormally(elapsedTime);
}

/**
 * @brief Writes output files.
 *
 * @details output files are written if the step is a multiple of the output frequency.
 *
 */
void Engine::writeOutput()
{
    _averagePhysicalData.updateAverages(_physicalData);

    const auto outputFrequency = settings::OutputFileSettings::getOutputFrequency();

    if (0 == _step % outputFrequency)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFrequency));

        const auto dt             = settings::TimingsSettings::getTimeStep();
        const auto step0          = _timings.getStepCount();
        const auto effectiveStep  = _step + step0;
        const auto simulationTime = static_cast<double>(effectiveStep) * dt * constants::_FS_TO_PS_;
        const auto loopTime       = _timings.calculateLoopTime(_step);

        _engineOutput.writeEnergyFile(effectiveStep, loopTime, _averagePhysicalData);
        _engineOutput.writeInstantEnergyFile(effectiveStep, loopTime, _physicalData);
        _engineOutput.writeMomentumFile(effectiveStep, _averagePhysicalData);
        _engineOutput.writeInfoFile(simulationTime, loopTime, _averagePhysicalData);
        _engineOutput.writeXyzFile(_simulationBox);
        _engineOutput.writeVelFile(_simulationBox);
        _engineOutput.writeForceFile(_simulationBox);
        _engineOutput.writeChargeFile(_simulationBox);
        _engineOutput.writeRstFile(_simulationBox, _step + step0);

        _engineOutput.writeVirialFile(effectiveStep, _physicalData);   // use physicalData instead of averagePhysicalData
        _engineOutput.writeStressFile(effectiveStep, _physicalData);   // use physicalData instead of averagePhysicalData
        _engineOutput.writeBoxFile(effectiveStep, _simulationBox.getBox());

        _averagePhysicalData = physicalData::PhysicalData();
    }

    _physicalData.reset();
}