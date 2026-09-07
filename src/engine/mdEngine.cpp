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
#include "globalTimer.hpp"
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "progressbar.hpp"          // for progressbar
#include "referencesOutput.hpp"     // for ReferencesOutput
#include "settings.hpp"             // for Settings
#include "timingsSettings.hpp"      // for TimingsSettings
#include "velocityVerlet.hpp"

using namespace engine;
using namespace out;
using namespace settings;
using namespace constants;
using namespace physicalData;

using virial::intraMolecularVirialCorrection;

/**
 * @brief Constructor for MDEngine
 *
 * @details This constructor initializes the MDEngine with default settings.
 */
MDEngine::MDEngine()
    : _integrator(std::make_unique<integrator::VelocityVerlet>()),
      _thermostat(std::make_unique<thermostat::Thermostat>()),
      _manostat(std::make_unique<manostat::Manostat>())
{
}

/**
 * @brief Run the simulation for numberOfSteps steps.
 *
 */
void MDEngine::run()
{
    _physicalData->calculateKinetics(getSimulationBox());

    _engineOutput.getLogOutput().writeInitialMomentum(
        norm(_physicalData->getMomentum())
    );

    _nSteps = TimingsSettings::getNumberOfSteps();
    progressbar bar(static_cast<int>(_nSteps), true, std::cout);

    for (; _step <= _nSteps; ++_step)
    {
        bar.update();
        takeStep();

        writeOutput();
        deleteTmpFiles();
    }

    timings::GlobalTimer::get().stopSimulationTimer();

    const auto elapsedTime =
        timings::GlobalTimer::get().calculateElapsedTime() * constants::MS_TO_S;

    references::ReferencesOutput::writeReferencesFile();

    _engineOutput.writeTimingsFile();

    _engineOutput.getLogOutput().writeEndedNormally(elapsedTime);
    _engineOutput.getStdoutOutput().writeEndedNormally(elapsedTime);
}

/**
 * @brief MD Loop before force calculation.
 *
 */
void MDEngine::takeStepBeforeForces()
{
    _thermostat->applyThermostatHalfStep(*_simulationBox, *_physicalData);

    if (_constraints->isMShakeActive())
        _simulationBox->updateOldPositions();

    _integrator->firstStep(*_simulationBox);

    _constraints->applyShake(*_simulationBox);
}

/**
 * @brief MD Loop after force calculation.
 *
 */
void MDEngine::takeStepAfterForces()
{
    _resetKinetics.resetForces(_step, *_simulationBox);

    _constraints->applyDistanceConstraints(
        *_simulationBox,
        *_physicalData,
        calculateTotalSimulationTime()
    );

    _constraints->calculateConstraintBondRefs(*_simulationBox);

    if (!Settings::isHybridJobtype())
    {
        const auto virial = intraMolecularVirialCorrection(*_simulationBox);
        _physicalData->addVirial(virial);
    }

    _thermostat->applyThermostatOnForces(*_simulationBox);

    _integrator->secondStep(*_simulationBox);

    _constraints->applyRattle(*_simulationBox);

    _thermostat->applyThermostat(*_simulationBox, *_physicalData);

    _physicalData->calculateKinetics(*_simulationBox);

    _manostat->applyManostat(*_simulationBox, *_physicalData);

    _resetKinetics.reset(_step, *_physicalData, *_simulationBox);

    _thermostat->applyTemperatureRamping();

    if (Settings::isQMOnlyJobtype())
    {
        const auto nQMAtoms = _simulationBox->getNumberOfQMAtoms();
        _physicalData->setNumberOfQMAtoms(static_cast<double>(nQMAtoms));
    }
}

void MDEngine::calculateForcesWrapper()
{
    _simulationBox->resetAllForces();
    calculateForces();
}

/**
 * @brief Takes one step in the simulation.
 *
 */
void MDEngine::takeStep()
{
    takeStepBeforeForces();

    calculateForcesWrapper();

    takeStepAfterForces();
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
    const auto outputFreq = OutputFileSettings::getOutputFrequency();
    const auto step0      = TimingsSettings::getStepCount();
    const auto effStep    = _step + step0;

    if (0 == _step % outputFreq)
    {
        _engineOutput.writeXyzFile(*_simulationBox, effStep);
        _engineOutput.writeVelFile(*_simulationBox, effStep);
        _engineOutput.writeForceFile(*_simulationBox, effStep);
        _engineOutput.writeChargeFile(*_simulationBox, effStep);
        _engineOutput.writeRstFile(*_simulationBox, *_thermostat, effStep);

        _engineOutput.writeVirialFile(
            effStep,
            *_physicalData
        );   // use physicalData instead of averagePhysicalData

        _engineOutput.writeStressFile(
            effStep,
            *_physicalData
        );   // use physicalData instead of averagePhysicalData

        _engineOutput.writeBoxFile(effStep, _simulationBox->getBox());

        if (Settings::isHybridJobtype())
            _engineOutput.writeHybridCenterXyzFile(_configurator, effStep);
    }

    // NOTE:
    // stop and restart immediately time manager - maximum lost time is en file
    // writing in last step of simulation but on the other hand setup is now
    // included in total simulation time
    // Unfortunately, setup is therefore included in the first looptime output
    // but this is not a big problem - could also be a feature and not a bug
    timings::GlobalTimer::get().stopAndRestartSimulationTimer();

    _physicalData->setLoopTime(timings::GlobalTimer::get().calculateLoopTime());
    _averagePhysicalData.updateAverages(*_physicalData);

    if (0 == _step % outputFreq)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFreq));

        const auto dt            = TimingsSettings::getTimeStep();
        const auto effStepDouble = static_cast<double>(effStep);
        const auto simTime       = effStepDouble * dt * FS_TO_PS;

        _engineOutput.writeEnergyFile(effStep, _averagePhysicalData);
        _engineOutput.writeInstantEnergyFile(effStep, *_physicalData);
        _engineOutput.writeInfoFile(simTime, _averagePhysicalData);
        _engineOutput.writeMomentumFile(effStep, _averagePhysicalData);

        _averagePhysicalData = PhysicalData();
    }

    _physicalData->reset();
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
 * @brief get the reference to the instant energy output
 *
 * @return out::EnergyOutput&
 */
out::EnergyOutput &MDEngine::getInstantEnergyOutput()
{
    return _engineOutput.getInstantEnergyOutput();
}

/**
 * @brief get the reference to the momentum output
 *
 * @return out::MomentumOutput&
 */
out::MomentumOutput &MDEngine::getMomentumOutput()
{
    return _engineOutput.getMomentumOutput();
}

/**
 * @brief get the reference to the xyz hybrid center output
 *
 * @return out::TrajectoryOutput&
 */
out::TrajectoryOutput &MDEngine::getXyzHybridCenterOutput()
{
    return _engineOutput.getXyzHybridCenterOutput();
}

/**
 * @brief get the reference to the vel output
 *
 * @return out::TrajectoryOutput&
 */
out::TrajectoryOutput &MDEngine::getVelOutput()
{
    return _engineOutput.getVelOutput();
}

/**
 * @brief get the reference to the charge output
 *
 * @return out::TrajectoryOutput&
 */
out::TrajectoryOutput &MDEngine::getChargeOutput()
{
    return _engineOutput.getChargeOutput();
}

/**
 * @brief get the reference to the virial output
 *
 * @return out::VirialOutput&
 */
out::VirialOutput &MDEngine::getVirialOutput()
{
    return _engineOutput.getVirialOutput();
}

/**
 * @brief get the reference to the stress output
 *
 * @return out::StressOutput&
 */
out::StressOutput &MDEngine::getStressOutput()
{
    return _engineOutput.getStressOutput();
}

/**
 * @brief get the reference to the box file output
 *
 * @return out::BoxFileOutput&
 */
out::BoxFileOutput &MDEngine::getBoxFileOutput()
{
    return _engineOutput.getBoxFileOutput();
}

/**
 * @brief get the reference to the ring polymer rst file output
 *
 * @return out::RingPolymerRestartFileOutput&
 */
RingPolymerRestartFileOutput &MDEngine::getRingPolymerRstFileOutput()
{
    return _engineOutput.getRingPolymerRstFileOutput();
}

/**
 * @brief get the reference to the ring polymer xyz output
 *
 * @return out::RingPolymerTrajectoryOutput&
 */
RingPolymerTrajectoryOutput &MDEngine::getRingPolymerXyzOutput()
{
    return _engineOutput.getRingPolymerXyzOutput();
}

/**
 * @brief get the reference to the ring polymer vel output
 *
 * @return out::RingPolymerTrajectoryOutput&
 */
RingPolymerTrajectoryOutput &MDEngine::getRingPolymerVelOutput()
{
    return _engineOutput.getRingPolymerVelOutput();
}

/**
 * @brief get the reference to the ring polymer force output
 *
 * @return out::RingPolymerTrajectoryOutput&
 */
RingPolymerTrajectoryOutput &MDEngine::getRingPolymerForceOutput()
{
    return _engineOutput.getRingPolymerForceOutput();
}

/**
 * @brief get the reference to the ring polymer charge output
 *
 * @return out::RingPolymerTrajectoryOutput&
 */
RingPolymerTrajectoryOutput &MDEngine::getRingPolymerChargeOutput()
{
    return _engineOutput.getRingPolymerChargeOutput();
}

/**
 * @brief get the reference to the ring polymer energy output
 *
 * @return out::RingPolymerEnergyOutput&
 */
RingPolymerEnergyOutput &MDEngine::getRingPolymerEnergyOutput()
{
    return _engineOutput.getRingPolymerEnergyOutput();
}
