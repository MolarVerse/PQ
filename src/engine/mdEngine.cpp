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

#include "mdEngine.hpp"

#include "constants/conversionFactors.hpp"   // for _FS_TO_PS_
#include "logOutput.hpp"                     // for LogOutput
#include "outputFileSettings.hpp"            // for OutputFileSettings
#include "progressbar.hpp"                   // for progressbar
#include "referencesOutput.hpp"              // for ReferencesOutput
#include "settings.hpp"                      // for Settings
#include "stdoutOutput.hpp"                  // for StdoutOutput
#include "timingsSettings.hpp"               // for TimingsSettings
#include "vector3d.hpp"                      // for norm

using namespace engine;

/**
 * @brief Run the simulation for numberOfSteps steps.
 *
 */
void MDEngine::run()
{
    _physicalData.calculateKinetics(getSimulationBox());

    _engineOutput.getLogOutput().writeInitialMomentum(
        norm(_physicalData.getMomentum())
    );

    const auto  numberOfSteps = settings::TimingsSettings::getNumberOfSteps();
    progressbar bar(static_cast<int>(numberOfSteps), true, std::cout);

    for (; _step <= numberOfSteps; ++_step)
    {
        bar.update();
        takeStep();

        writeOutput();
    }

    _timer.stopSimulationTimer();

    const auto elapsedTime = double(_timer.calculateElapsedTime()) * 1e-3;

    _engineOutput.setTimerName("Output");
    _timer.addTimer(_engineOutput.getTimer());

    _thermostat->setTimerName("Thermostat");
    _timer.addTimer(_thermostat->getTimer());

    _integrator->setTimerName("Integrator");
    _timer.addTimer(_integrator->getTimer());

    _constraints.setTimerName("Constraints");
    _timer.addTimer(_constraints.getTimer());

    _cellList.setTimerName("Cell List");
    _timer.addTimer(_cellList.getTimer());

    _potential->setTimerName("Potential");
    _timer.addTimer(_potential->getTimer());

    _intraNonBonded.setTimerName("IntraNonBonded");
    _timer.addTimer(_intraNonBonded.getTimer());

    _virial->setTimerName("Virial");
    _timer.addTimer(_virial->getTimer());

    _physicalData.setTimerName("Physical Data");
    _timer.addTimer(_physicalData.getTimer());

    _manostat->setTimerName("Manostat");
    _timer.addTimer(_manostat->getTimer());

    _resetKinetics.setTimerName("Reset Kinetics");
    _timer.addTimer(_resetKinetics.getTimer());

#ifdef WITH_CUDA
    _cudaPotential.setTimerName("Cuda Potential");
    _timer.addTimer(_cudaPotential.getTimer());
#endif

    references::ReferencesOutput::writeReferencesFile();

    _engineOutput.writeTimingsFile(_timer);

    _engineOutput.getLogOutput().writeEndedNormally(elapsedTime);
    _engineOutput.getStdoutOutput().writeEndedNormally(elapsedTime);
}

/**
 * @brief Writes output files.
 *
 * @details output files are written if the step is a multiple of the output
 * frequency.
 *
 */
void MDEngine::writeOutput()
{
    const auto outputFreq = settings::OutputFileSettings::getOutputFrequency();
    const auto step0      = settings::TimingsSettings::getStepCount();
    const auto effStep    = _step + step0;

    if (0 == _step % outputFreq)
    {
        _engineOutput.writeXyzFile(_simulationBox);
        _engineOutput.writeVelFile(_simulationBox);
        _engineOutput.writeForceFile(_simulationBox);
        _engineOutput.writeChargeFile(_simulationBox);
        _engineOutput.writeRstFile(_simulationBox, _step + step0);

        _engineOutput.writeVirialFile(
            effStep,
            _physicalData
        );   // use physicalData instead of averagePhysicalData

        _engineOutput.writeStressFile(
            effStep,
            _physicalData
        );   // use physicalData instead of averagePhysicalData

        _engineOutput.writeBoxFile(effStep, _simulationBox.getBox());
    }

    // NOTE:
    // stop and restart immediately time manager - maximum lost time is en file
    // writing in last step of simulation but on the other hand setup is now
    // included in total simulation time
    // Unfortunately, setup is therefore included in the first looptime output
    // but this is not a big problem - could also be a feature and not a bug
    _timer.stopSimulationTimer();
    _timer.startSimulationTimer();

    _physicalData.setLoopTime(_timer.calculateLoopTime());
    _averagePhysicalData.updateAverages(_physicalData);

    if (0 == _step % outputFreq)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFreq));

        const auto dt            = settings::TimingsSettings::getTimeStep();
        const auto effStepDouble = static_cast<double>(effStep);
        const auto simTime       = effStepDouble * dt * constants::_FS_TO_PS_;

        _engineOutput.writeEnergyFile(effStep, _averagePhysicalData);
        _engineOutput.writeInstantEnergyFile(effStep, _physicalData);
        _engineOutput.writeInfoFile(simTime, _averagePhysicalData);
        _engineOutput.writeMomentumFile(effStep, _averagePhysicalData);

        _averagePhysicalData = physicalData::PhysicalData();
    }

    _physicalData.reset();
}

/**
 * @brief get the reference to the reset kinetics
 *
 * @return resetKinetics::ResetKinetics&
 */
resetKinetics::ResetKinetics &MDEngine::getResetKinetics()
{
    return _resetKinetics;
}

/**
 * @brief get the reference to the integrator
 *
 * @return integrator::Integrator&
 */
integrator::Integrator &MDEngine::getIntegrator() { return *_integrator; }

/**
 * @brief get the reference to the thermostat
 *
 * @return thermostat::Thermostat&
 */
thermostat::Thermostat &MDEngine::getThermostat() { return *_thermostat; }

/**
 * @brief get the reference to the manostat
 *
 * @return manostat::Manostat&
 */
manostat::Manostat &MDEngine::getManostat() { return *_manostat; }

/**
 * @brief get the reference to the energy output
 *
 * @return output::EnergyOutput&
 */
output::EnergyOutput &MDEngine::getEnergyOutput()
{
    return _engineOutput.getEnergyOutput();
}

/**
 * @brief get the reference to the instant energy output
 *
 * @return output::EnergyOutput&
 */
output::EnergyOutput &MDEngine::getInstantEnergyOutput()
{
    return _engineOutput.getInstantEnergyOutput();
}

/**
 * @brief get the reference to the momentum output
 *
 * @return output::MomentumOutput&
 */
output::MomentumOutput &MDEngine::getMomentumOutput()
{
    return _engineOutput.getMomentumOutput();
}

/**
 * @brief get the reference to the xyz output
 *
 * @return output::TrajectoryOutput&
 */
output::TrajectoryOutput &MDEngine::getXyzOutput()
{
    return _engineOutput.getXyzOutput();
}

/**
 * @brief get the reference to the vel output
 *
 * @return output::TrajectoryOutput&
 */
output::TrajectoryOutput &MDEngine::getVelOutput()
{
    return _engineOutput.getVelOutput();
}

/**
 * @brief get the reference to the force output
 *
 * @return output::TrajectoryOutput&
 */
output::TrajectoryOutput &MDEngine::getForceOutput()
{
    return _engineOutput.getForceOutput();
}

/**
 * @brief get the reference to the charge output
 *
 * @return output::TrajectoryOutput&
 */
output::TrajectoryOutput &MDEngine::getChargeOutput()
{
    return _engineOutput.getChargeOutput();
}

/**
 * @brief get the reference to the rst file output
 *
 * @return output::RstFileOutput&
 */
output::RstFileOutput &MDEngine::getRstFileOutput()
{
    return _engineOutput.getRstFileOutput();
}

/**
 * @brief get the reference to the info output
 *
 * @return output::InfoOutput&
 */
output::InfoOutput &MDEngine::getInfoOutput()
{
    return _engineOutput.getInfoOutput();
}

/**
 * @brief get the reference to the virial output
 *
 * @return output::VirialOutput&
 */
output::VirialOutput &MDEngine::getVirialOutput()
{
    return _engineOutput.getVirialOutput();
}

/**
 * @brief get the reference to the stress output
 *
 * @return output::StressOutput&
 */
output::StressOutput &MDEngine::getStressOutput()
{
    return _engineOutput.getStressOutput();
}

/**
 * @brief get the reference to the box file output
 *
 * @return output::BoxFileOutput&
 */
output::BoxFileOutput &MDEngine::getBoxFileOutput()
{
    return _engineOutput.getBoxFileOutput();
}

/**
 * @brief get the reference to the ring polymer rst file output
 *
 * @return output::RingPolymerRestartFileOutput&
 */
RPMDRestartFileOutput &MDEngine::getRingPolymerRstFileOutput()
{
    return _engineOutput.getRingPolymerRstFileOutput();
}

/**
 * @brief get the reference to the ring polymer xyz output
 *
 * @return output::RingPolymerTrajectoryOutput&
 */
RPMDTrajectoryOutput &MDEngine::getRingPolymerXyzOutput()
{
    return _engineOutput.getRingPolymerXyzOutput();
}

/**
 * @brief get the reference to the ring polymer vel output
 *
 * @return output::RingPolymerTrajectoryOutput&
 */
RPMDTrajectoryOutput &MDEngine::getRingPolymerVelOutput()
{
    return _engineOutput.getRingPolymerVelOutput();
}

/**
 * @brief get the reference to the ring polymer force output
 *
 * @return output::RingPolymerTrajectoryOutput&
 */
RPMDTrajectoryOutput &MDEngine::getRingPolymerForceOutput()
{
    return _engineOutput.getRingPolymerForceOutput();
}

/**
 * @brief get the reference to the ring polymer charge output
 *
 * @return output::RingPolymerTrajectoryOutput&
 */
RPMDTrajectoryOutput &MDEngine::getRingPolymerChargeOutput()
{
    return _engineOutput.getRingPolymerChargeOutput();
}

/**
 * @brief get the reference to the ring polymer energy output
 *
 * @return output::RingPolymerEnergyOutput&
 */
RPMDEnergyOutput &MDEngine::getRingPolymerEnergyOutput()
{
    return _engineOutput.getRingPolymerEnergyOutput();
}