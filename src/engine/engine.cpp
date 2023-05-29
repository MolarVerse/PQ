#include "engine.hpp"
#include "constants.hpp"

using namespace std;

void Engine::run()
{
    getSimulationBox().calculateDegreesOfFreedom();

    _physicalData.calculateKineticEnergyAndMomentum(getSimulationBox());

    _logOutput->writeInitialMomentum(_physicalData.getMomentum());
    _stdoutOutput->writeInitialMomentum(_physicalData.getMomentum());

    for (int i = 0; i < _timings.getNumberOfSteps(); i++)
    {
        _physicalData.resetAverageData();

        takeStep();

        _physicalData.addAverageTemperature(_thermostat->getTemperature());
    }
}

void Engine::takeStep()
{
    _integrator->firstStep(_simulationBox, _timings);

    _cellList.updateCellList(_simulationBox);

    _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _integrator->secondStep(_simulationBox, _timings);

    _thermostat->applyThermostat(_simulationBox);

    _physicalData.calculateKineticEnergyAndMomentum(_simulationBox);
}
