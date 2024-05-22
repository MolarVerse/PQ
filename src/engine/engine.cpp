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
#include "settings.hpp"                      // for Settings
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

#ifdef WITH_KOKKOS
    _kokkosVelocityVerlet.setTimerName("Kokkos Velocity Verlet");
    _timer.addTimer(_kokkosVelocityVerlet.getTimer());

    _kokkosPotential.setTimerName("Kokkos Potential");
    _timer.addTimer(_kokkosPotential.getTimer());
#endif

    references::ReferencesOutput::writeReferencesFile();

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
void Engine::writeOutput()
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
 * @brief Adds a timings section to the timingsSection vector.
 *
 */
void Engine::addTimer(const timings::Timer &timings)
{
    _timer.addTimer(timings);
}

/**
 * @brief checks if the force field is activated
 *
 * @return true
 * @return false
 */
bool Engine::isForceFieldNonCoulombicsActivated() const
{
    return _forceField.isNonCoulombicActivated();
}

/**
 * @brief checks if the guff formalism is activated
 *
 * @return true
 * @return false
 */
bool Engine::isGuffActivated() const
{
    return !_forceField.isNonCoulombicActivated();
}

/**
 * @brief checks if the cell list is activated
 *
 * @return true
 * @return false
 */
bool Engine::isCellListActivated() const { return _cellList.isActive(); }

/**
 * @brief checks if any constraints are activated
 *
 * @return true
 * @return false
 */
bool Engine::isConstraintsActivated() const { return _constraints.isActive(); }

/**
 * @brief checks if the intra non bonded interactions are activated
 *
 * @return true
 * @return false
 */
bool Engine::isIntraNonBondedActivated() const
{
    return _intraNonBonded.isActive();
}

/**
 * @brief get the reference to the cell list
 *
 * @return simulationBox::CellList&
 */
simulationBox::CellList &Engine::getCellList() { return _cellList; }

/**
 * @brief get the reference to the simulation box
 *
 * @return simulationBox::SimulationBox&
 */
simulationBox::SimulationBox &Engine::getSimulationBox()
{
    return _simulationBox;
}

/**
 * @brief get the reference to the physical data
 *
 * @return physicalData::PhysicalData&
 */
physicalData::PhysicalData &Engine::getPhysicalData() { return _physicalData; }

/**
 * @brief get the reference to the average physical data
 *
 * @return physicalData::PhysicalData&
 */
physicalData::PhysicalData &Engine::getAveragePhysicalData()
{
    return _averagePhysicalData;
}

/**
 * @brief get the reference to the Constraints
 *
 * @return timings::Timer&
 */
constraints::Constraints &Engine::getConstraints() { return _constraints; }

/**
 * @brief get the reference to the force field
 *
 * @return forceField::ForceField&
 */
forceField::ForceField &Engine::getForceField() { return _forceField; }

/**
 * @brief get the reference to the intra non bonded interactions
 *
 * @return intraNonBonded::IntraNonBonded&
 */
intraNonBonded::IntraNonBonded &Engine::getIntraNonBonded()
{
    return _intraNonBonded;
}

/**
 * @brief get the reference to the reset kinetics
 *
 * @return resetKinetics::ResetKinetics&
 */
resetKinetics::ResetKinetics &Engine::getResetKinetics()
{
    return _resetKinetics;
}

/**
 * @brief get the reference to the virial
 *
 * @return thermostat::Thermostat&
 */
virial::Virial &Engine::getVirial() { return *_virial; }

/**
 * @brief get the reference to the integrator
 *
 * @return thermostat::Thermostat&
 */
integrator::Integrator &Engine::getIntegrator() { return *_integrator; }

/**
 * @brief get the reference to the potential
 *
 * @return thermostat::Thermostat&
 */
potential::Potential &Engine::getPotential() { return *_potential; }

/**
 * @brief get the reference to the thermostat
 *
 * @return thermostat::Thermostat&
 */
thermostat::Thermostat &Engine::getThermostat() { return *_thermostat; }

/**
 * @brief get the reference to the manostat
 *
 * @return thermostat::Thermostat&
 */
manostat::Manostat &Engine::getManostat() { return *_manostat; }

/**
 * @brief get the reference to the engine output
 *
 * @return thermostat::Thermostat&
 */
EngineOutput &Engine::getEngineOutput() { return _engineOutput; }

/**
 * @brief get the reference to the energy output
 *
 * @return timings::Timer&
 */
output::EnergyOutput &Engine::getEnergyOutput()
{
    return _engineOutput.getEnergyOutput();
}

/**
 * @brief get the reference to the instant energy output
 *
 * @return timings::Timer&
 */
output::EnergyOutput &Engine::getInstantEnergyOutput()
{
    return _engineOutput.getInstantEnergyOutput();
}

/**
 * @brief get the reference to the momentum output
 *
 * @return timings::Timer&
 */
output::MomentumOutput &Engine::getMomentumOutput()
{
    return _engineOutput.getMomentumOutput();
}

/**
 * @brief get the reference to the xyz output
 *
 * @return timings::Timer&
 */
output::TrajectoryOutput &Engine::getXyzOutput()
{
    return _engineOutput.getXyzOutput();
}

/**
 * @brief get the reference to the vel output
 *
 * @return timings::Timer&
 */
output::TrajectoryOutput &Engine::getVelOutput()
{
    return _engineOutput.getVelOutput();
}

/**
 * @brief get the reference to the force output
 *
 * @return timings::Timer&
 */
output::TrajectoryOutput &Engine::getForceOutput()
{
    return _engineOutput.getForceOutput();
}

/**
 * @brief get the reference to the charge output
 *
 * @return timings::Timer&
 */
output::TrajectoryOutput &Engine::getChargeOutput()
{
    return _engineOutput.getChargeOutput();
}

/**
 * @brief get the reference to the log output
 *
 * @return timings::Timer&
 */
output::LogOutput &Engine::getLogOutput()
{
    return _engineOutput.getLogOutput();
}

/**
 * @brief get the reference to the stdout output
 *
 * @return timings::Timer&
 */
output::StdoutOutput &Engine::getStdoutOutput()
{
    return _engineOutput.getStdoutOutput();
}

/**
 * @brief get the reference to the rst file output
 *
 * @return timings::Timer&
 */
output::RstFileOutput &Engine::getRstFileOutput()
{
    return _engineOutput.getRstFileOutput();
}

/**
 * @brief get the reference to the info output
 *
 * @return timings::Timer&
 */
output::InfoOutput &Engine::getInfoOutput()
{
    return _engineOutput.getInfoOutput();
}

/**
 * @brief get the reference to the virial output
 *
 * @return timings::Timer&
 */
output::VirialOutput &Engine::getVirialOutput()
{
    return _engineOutput.getVirialOutput();
}

/**
 * @brief get the reference to the stress output
 *
 * @return timings::Timer&
 */
output::StressOutput &Engine::getStressOutput()
{
    return _engineOutput.getStressOutput();
}

/**
 * @brief get the reference to the box file output
 *
 * @return timings::Timer&
 */
output::BoxFileOutput &Engine::getBoxFileOutput()
{
    return _engineOutput.getBoxFileOutput();
}

/**
 * @brief get the reference to the ring polymer rst file output
 *
 * @return timings::Timer&
 */
RPRestartFileOutput &Engine::getRingPolymerRstFileOutput()
{
    return _engineOutput.getRingPolymerRstFileOutput();
}

/**
 * @brief get the reference to the ring polymer xyz output
 *
 * @return timings::Timer&
 */
RPTrajectoryOutput &Engine::getRingPolymerXyzOutput()
{
    return _engineOutput.getRingPolymerXyzOutput();
}

/**
 * @brief get the reference to the ring polymer vel output
 *
 * @return timings::Timer&
 */
RPTrajectoryOutput &Engine::getRingPolymerVelOutput()
{
    return _engineOutput.getRingPolymerVelOutput();
}

/**
 * @brief get the reference to the ring polymer force output
 *
 * @return timings::Timer&
 */
RPTrajectoryOutput &Engine::getRingPolymerForceOutput()
{
    return _engineOutput.getRingPolymerForceOutput();
}

/**
 * @brief get the reference to the ring polymer charge output
 *
 * @return timings::Timer&
 */
RPTrajectoryOutput &Engine::getRingPolymerChargeOutput()
{
    return _engineOutput.getRingPolymerChargeOutput();
}

/**
 * @brief get the reference to the ring polymer energy output
 *
 * @return timings::Timer&
 */
RPEnergyOutput &Engine::getRingPolymerEnergyOutput()
{
    return _engineOutput.getRingPolymerEnergyOutput();
}

/**
 * @brief get the pointer to the force field
 *
 * @return thermostat::Thermostat&
 */
forceField::ForceField *Engine::getForceFieldPtr() { return &_forceField; }
