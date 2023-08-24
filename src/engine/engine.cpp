#include "engine.hpp"

#include "constants.hpp"
#include "progressbar.hpp"

#include <iostream>

using namespace std;
using namespace simulationBox;
using namespace physicalData;
using namespace settings;
using namespace timings;
using namespace engine;
using namespace output;

void Engine::run()
{
    _timings.beginTimer();

    _simulationBox.calculateDegreesOfFreedom();
    _simulationBox.calculateCenterOfMassMolecules();

    _physicalData.calculateKineticEnergyAndMomentum(getSimulationBox());

    _engineOutput.getLogOutput().writeInitialMomentum(_physicalData.getMomentum());
    _engineOutput.getStdoutOutput().writeInitialMomentum(_physicalData.getMomentum());

    const auto  numberOfSteps = _timings.getNumberOfSteps();
    progressbar bar(static_cast<int>(numberOfSteps));

    for (; _step <= numberOfSteps; ++_step)
    {
        bar.update();
        takeStep();

        writeOutput();
    }

    _timings.endTimer();

    cout << '\n' << '\n';

    cout << "Total time: " << double(_timings.calculateElapsedTime()) * 1e-3 << "s" << '\n';
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
 *  5.  Calculate constraint bond references
 *  6.  Second step of the integrator
 *  7.  Apply RATTLE
 *  8.  Apply thermostat
 *  9.  Calculate kinetic energy and momentum
 * 10.  Calculate virial
 * 11.  Apply manostat
 * 12.  Reset temperature and momentum
 *
 */
void Engine::takeStep()
{
    _integrator->firstStep(_simulationBox);

    _constraints.applyShake(_simulationBox);

    _cellList.updateCellList(_simulationBox);

    // _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _intraNonBonded.calculate(_simulationBox, _physicalData);

    _forceField.calculateBondedInteractions(_simulationBox, _physicalData);

    _constraints.calculateConstraintBondRefs(_simulationBox);

    _integrator->secondStep(_simulationBox);

    _constraints.applyRattle();

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKineticEnergyAndMomentum(_simulationBox);

    _virial->calculateVirial(_simulationBox, _physicalData);

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

    const auto outputFrequency = Output::getOutputFrequency();

    if (0 == _step % outputFrequency)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFrequency));

        const auto dt             = _timings.getTimestep();
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

        cout << '\n' << '\n';

        cout << "Coulomb energy: " << _averagePhysicalData.getCoulombEnergy() << '\n';
        cout << "Non Coulomb energy: " << _averagePhysicalData.getNonCoulombEnergy() << '\n';
        cout << "intra coulomb energy " << _averagePhysicalData.getIntraCoulombEnergy() << '\n';
        cout << "intra non coulomb energy " << _averagePhysicalData.getIntraNonCoulombEnergy() << '\n';
        cout << "bond energy " << _averagePhysicalData.getBondEnergy() << '\n';
        cout << "angle energy " << _averagePhysicalData.getAngleEnergy() << '\n';
        cout << "dihedral energy " << _averagePhysicalData.getDihedralEnergy() << '\n';
        cout << "improper energy " << _averagePhysicalData.getImproperEnergy() << '\n';
        cout << "Kinetic energy: " << _averagePhysicalData.getKineticEnergy() << '\n';
        cout << '\n';

        cout << "Temperature: " << _averagePhysicalData.getTemperature() << '\n';
        cout << "Momentum: " << _averagePhysicalData.getMomentum() << '\n';
        cout << '\n';

        cout << "Volume: " << _averagePhysicalData.getVolume() << '\n';
        cout << "Density: " << _averagePhysicalData.getDensity() << '\n';
        cout << "Pressure: " << _averagePhysicalData.getPressure() << '\n';

        cout << '\n' << '\n';

        _averagePhysicalData = PhysicalData();
    }
}