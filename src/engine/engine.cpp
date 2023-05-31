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

    for (int i = 0; i < _timings.getNumberOfSteps(); i++)
    {
        _averagePhysicalData = PhysicalData();

        takeStep();

        _averagePhysicalData.updateAverages(_physicalData);
    }

    cout << "Couloumb energy: " << _averagePhysicalData.getCoulombEnergy() << endl;
    cout << "Non Couloumb energy: " << _averagePhysicalData.getNonCoulombEnergy() << endl;
    cout << "Kinetic energy: " << _averagePhysicalData.getKineticEnergy() << endl;

    cout << "Temperature: " << _averagePhysicalData.getTemperature() << endl;
    cout << "Momentum: " << _averagePhysicalData.getMomentum() << endl;

    cout << "Volume: " << _averagePhysicalData.getVolume() << endl;
    cout << "Density: " << _averagePhysicalData.getDensity() << endl;
    cout << "Pressure: " << _averagePhysicalData.getPressure() << endl;

    cout << endl
         << endl;
}

void Engine::takeStep()
{
    _integrator->firstStep(_simulationBox, _timings);

    _cellList.updateCellList(_simulationBox);

    _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _integrator->secondStep(_simulationBox, _timings);

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKineticEnergyAndMomentum(_simulationBox);

    _virial->calculateVirial(_simulationBox, _physicalData);

    _manostat->calculatePressure(*_virial, _physicalData);
}
