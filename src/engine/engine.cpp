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
    }
}

void Engine::takeStep()
{
    _integrator->firstStep(getSimulationBox(), _timings);

    if (_cellList.isActivated())
    {
        _cellList.updateCellList(getSimulationBox());
    }

    _potential->calculateForces(getSimulationBox(), _physicalData, _cellList);

    _integrator->secondStep(getSimulationBox(), _timings);

    _thermostat->applyThermostat(getSimulationBox());

    _physicalData.calculateKineticEnergyAndMomentum(getSimulationBox());

    _physicalData.addAverageTemperature(_thermostat->getTemperature());
}
