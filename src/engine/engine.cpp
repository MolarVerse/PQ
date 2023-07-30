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

    _logOutput->writeInitialMomentum(_physicalData.getMomentum());
    _stdoutOutput->writeInitialMomentum(_physicalData.getMomentum());

    const auto  numberOfSteps = _timings.getNumberOfSteps();
    progressbar bar(static_cast<int>(numberOfSteps));

    for (; _step <= numberOfSteps; ++_step)
    {
        bar.update();
        takeStep();

        writeOutput();
    }

    _timings.endTimer();

    cout << endl << endl;

    cout << "Total time: " << static_cast<int>(_timings.calculateElapsedTime()) * 1e-6 << endl;

    cout << endl << endl;

    cout << "Coulomb energy: " << _physicalData.getCoulombEnergy() << endl;
    cout << "Non Coulomb energy: " << _physicalData.getNonCoulombEnergy() << endl;
    cout << "Kinetic energy: " << _physicalData.getKineticEnergy() << endl;

    cout << "Temperature: " << _physicalData.getTemperature() << endl;
    cout << "Momentum: " << _physicalData.getMomentum() << endl;

    cout << "Volume: " << _physicalData.getVolume() << endl;
    cout << "Density: " << _physicalData.getDensity() << endl;
    cout << "Pressure: " << _physicalData.getPressure() << endl;

    cout << endl << endl;
}

void Engine::takeStep()
{
    _integrator->firstStep(_simulationBox);

    _constraints.applyShake(_simulationBox);

    _cellList.updateCellList(_simulationBox);

    _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _constraints.calculateConstraintBondRefs(_simulationBox);

    _integrator->secondStep(_simulationBox);

    _constraints.applyRattle();

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKineticEnergyAndMomentum(_simulationBox);

    _virial->calculateVirial(_simulationBox, _physicalData);

    _manostat->applyManostat(_simulationBox, _physicalData);

    _resetKinetics->reset(_step, _physicalData, _simulationBox);
}

void Engine::writeOutput()
{
    _averagePhysicalData.updateAverages(_physicalData);

    const auto outputFrequency = Output::getOutputFrequency();

    if (_step % outputFrequency == 0)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFrequency));

        const auto dt             = _timings.getTimestep();
        const auto step0          = _timings.getStepCount();
        const auto effectiveStep  = _step + step0;
        const auto simulationTime = static_cast<double>(effectiveStep) * dt * constants::_FS_TO_PS_;

        _energyOutput->write(effectiveStep, _averagePhysicalData);
        _infoOutput->write(simulationTime, _averagePhysicalData);
        _xyzOutput->writeXyz(_simulationBox);
        _velOutput->writeVelocities(_simulationBox);
        _forceOutput->writeForces(_simulationBox);
        _chargeOutput->writeCharges(_simulationBox);
        _rstFileOutput->write(_simulationBox, _step + step0);

        _averagePhysicalData = PhysicalData();
    }
    else
    {
        // calclooptime
    }
}