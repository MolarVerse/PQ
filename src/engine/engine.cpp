#include "engine.hpp"

#include "constants.hpp"         // for _FS_TO_PS_
#include "logOutput.hpp"         // for LogOutput
#include "output.hpp"            // for Output
#include "progressbar.hpp"       // for progressbar
#include "stdoutOutput.hpp"      // for StdoutOutput
#include "timingsSettings.hpp"   // for TimingsSettings

#include <iostream>   // for operator<<, cout, ostream, basic_ostream

using namespace engine;

/**
 * @brief Run the simulation for numberOfSteps steps.
 *
 */
void Engine::run()
{
    _timings.beginTimer();

    _simulationBox.calculateDegreesOfFreedom();
    _simulationBox.calculateCenterOfMassMolecules();

    _physicalData.calculateKineticEnergyAndMomentum(getSimulationBox());

    _engineOutput.getLogOutput().writeInitialMomentum(_physicalData.getMomentum());
    _engineOutput.getStdoutOutput().writeInitialMomentum(_physicalData.getMomentum());

    const auto  numberOfSteps = settings::TimingsSettings::getNumberOfSteps();
    progressbar bar(static_cast<int>(numberOfSteps));

    for (; _step <= numberOfSteps; ++_step)
    {
        bar.update();
        takeStep();

        writeOutput();
    }

    _timings.endTimer();

    std::cout << '\n' << '\n';
    std::cout << "Total time: " << double(_timings.calculateElapsedTime()) * 1e-3 << "s" << '\n';
}

/**
 * @brief Takes one step in the simulation.
 *
 * @details The step is taken in the following order:
 *  1.  First step of the integrator
 *  2.  Apply SHAKE
 *  3.  Update cell list
 *  4.1 Calculate forces
 *  4.2 Calculate intra non bonded forces
 *  5.  Calculate virial
 *  6.  Calculate constraint bond references
 *  7.  Second step of the integrator
 *  8.  Apply RATTLE
 *  9.  Apply thermostat
 * 10.  Calculate kinetic energy and momentum
 * 11.  Apply manostat
 * 12.  Reset temperature and momentum
 *
 */
void Engine::takeStep()
{
    _integrator->firstStep(_simulationBox);

    _constraints.applyShake(_simulationBox);

    _cellList.updateCellList(_simulationBox);

    _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _intraNonBonded.calculate(_simulationBox, _physicalData);

    _virial->calculateVirial(_simulationBox, _physicalData);

    _forceField.calculateBondedInteractions(_simulationBox, _physicalData);

    _constraints.calculateConstraintBondRefs(_simulationBox);

    _integrator->secondStep(_simulationBox);

    _constraints.applyRattle();

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKineticEnergyAndMomentum(_simulationBox);

    _manostat->applyManostat(_simulationBox, _physicalData);

    _resetKinetics->reset(_step, _physicalData, _simulationBox);
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
    _physicalData.clearData();

    const auto outputFrequency = output::Output::getOutputFrequency();

    if (0 == _step % outputFrequency)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFrequency));

        const auto dt             = settings::TimingsSettings::getTimeStep();
        const auto step0          = _timings.getStepCount();
        const auto effectiveStep  = _step + step0;
        const auto simulationTime = static_cast<double>(effectiveStep) * dt * constants::_FS_TO_PS_;

        _engineOutput.writeEnergyFile(effectiveStep, _averagePhysicalData);
        _engineOutput.writeInfoFile(simulationTime, _averagePhysicalData);
        _engineOutput.writeXyzFile(_simulationBox);
        _engineOutput.writeVelFile(_simulationBox);
        _engineOutput.writeForceFile(_simulationBox);
        _engineOutput.writeChargeFile(_simulationBox);
        _engineOutput.writeRstFile(_simulationBox, _step + step0);

        if (_step == settings::TimingsSettings::getNumberOfSteps())
        {
            std::cout << '\n' << '\n';

            std::cout << "Coulomb energy: " << _averagePhysicalData.getCoulombEnergy() << '\n';
            std::cout << "Non Coulomb energy: " << _averagePhysicalData.getNonCoulombEnergy() << '\n';
            std::cout << "intra coulomb energy " << _averagePhysicalData.getIntraCoulombEnergy() << '\n';
            std::cout << "intra non coulomb energy " << _averagePhysicalData.getIntraNonCoulombEnergy() << '\n';
            std::cout << "bond energy " << _averagePhysicalData.getBondEnergy() << '\n';
            std::cout << "angle energy " << _averagePhysicalData.getAngleEnergy() << '\n';
            std::cout << "dihedral energy " << _averagePhysicalData.getDihedralEnergy() << '\n';
            std::cout << "improper energy " << _averagePhysicalData.getImproperEnergy() << '\n';
            std::cout << "Kinetic energy: " << _averagePhysicalData.getKineticEnergy() << '\n';
            std::cout << '\n';

            std::cout << "Temperature: " << _averagePhysicalData.getTemperature() << '\n';
            std::cout << "Momentum: " << _averagePhysicalData.getMomentum() << '\n';
            std::cout << '\n';

            std::cout << "Volume: " << _averagePhysicalData.getVolume() << '\n';
            std::cout << "Density: " << _averagePhysicalData.getDensity() << '\n';
            std::cout << "Pressure: " << _averagePhysicalData.getPressure() << '\n';

            std::cout << '\n' << '\n';
        }

        _averagePhysicalData = physicalData::PhysicalData();
    }
}