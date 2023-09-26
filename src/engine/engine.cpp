#include "engine.hpp"

#include "constants.hpp"            // for _FS_TO_PS_
#include "logOutput.hpp"            // for LogOutput
#include "output.hpp"               // for Output
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "progressbar.hpp"          // for progressbar
#include "stdoutOutput.hpp"         // for StdoutOutput
#include "timingsSettings.hpp"      // for TimingsSettings

#include <iostream>   // for operator<<, cout, ostream, basic_ostream

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
    _engineOutput.getStdoutOutput().writeInitialMomentum(norm(_physicalData.getMomentum()));

    const auto  numberOfSteps = settings::TimingsSettings::getNumberOfSteps();
    progressbar bar(static_cast<int>(numberOfSteps));

    for (; _step <= numberOfSteps; ++_step)
    {
        bar.update();
        takeStep();

        writeOutput();
    }

    _timings.endTimer();

    const auto elapsedTime = _timings.calculateElapsedTime() * 1e-3;

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
    _physicalData.reset();

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
        _engineOutput.writeMomentumFile(effectiveStep, _averagePhysicalData);
        _engineOutput.writeInfoFile(simulationTime, loopTime, _averagePhysicalData);
        _engineOutput.writeXyzFile(_simulationBox);
        _engineOutput.writeVelFile(_simulationBox);
        _engineOutput.writeForceFile(_simulationBox);
        _engineOutput.writeChargeFile(_simulationBox);
        _engineOutput.writeRstFile(_simulationBox, _step + step0);

        _averagePhysicalData = physicalData::PhysicalData();
    }
}