#include "engine.hpp"
#include "constants.hpp"

#include <iostream>

using namespace std;

void Engine::run()
{
    _simulationBox.calculateDegreesOfFreedom();
    _simulationBox.calculateCenterOfMassMolecules();

    _physicalData.calculateKineticEnergyAndMomentum(getSimulationBox());

    _logOutput->writeInitialMomentum(_physicalData.getMomentum());
    _stdoutOutput->writeInitialMomentum(_physicalData.getMomentum());

    const auto numberOfSteps = _timings.getNumberOfSteps();

    for (; _step <= numberOfSteps; ++_step)
    {
        takeStep();

        writeOutput();
    }

    cout << "Couloumb energy: " << _physicalData.getCoulombEnergy() << endl;
    cout << "Non Couloumb energy: " << _physicalData.getNonCoulombEnergy() << endl;
    cout << "Kinetic energy: " << _physicalData.getKineticEnergy() << endl;

    cout << "Temperature: " << _physicalData.getTemperature() << endl;
    cout << "Momentum: " << _physicalData.getMomentum() << endl;

    cout << "Volume: " << _physicalData.getVolume() << endl;
    cout << "Density: " << _physicalData.getDensity() << endl;
    cout << "Pressure: " << _physicalData.getPressure() << endl;

    cout << endl
         << endl;
}

void Engine::takeStep()
{
    _integrator->firstStep(_simulationBox);

    _cellList.updateCellList(_simulationBox);

    _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _integrator->secondStep(_simulationBox);

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKineticEnergyAndMomentum(_simulationBox);

    _virial->calculateVirial(_simulationBox, _physicalData);

    _manostat->calculatePressure(_physicalData);
}

void Engine::writeOutput()
{
    _averagePhysicalData.updateAverages(_physicalData);
    const auto outputFrequency = Output::getOutputFrequency();

    if (_step % outputFrequency == 0)
    {
        _averagePhysicalData.makeAverages(static_cast<double>(outputFrequency));

        _energyOutput->write(_step, _step0, _averagePhysicalData);

        _averagePhysicalData = PhysicalData();
    }
}