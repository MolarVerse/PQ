#include "thermostat.hpp"
#include "constants.hpp"

#include <cmath>

using namespace std;

void Thermostat::calculateTemperature(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    auto temperature = 0.0;

    for (const auto &molecule : simulationBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto velocities = molecule.getAtomVelocity(i);
            const auto mass = molecule.getAtomMass(i);

            temperature += mass * normSquared(velocities);
        }
    }

    temperature *= _TEMPERATURE_FACTOR_ / simulationBox.getDegreesOfFreedom();

    _temperature = temperature;
    physicalData.setTemperature(_temperature);
}

void Thermostat::applyThermostat(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    calculateTemperature(simulationBox, physicalData);
}

void BerendsenThermostat::applyThermostat(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    calculateTemperature(simulationBox, physicalData);

    const auto berendsenFactor = sqrt(1.0 + _timestep / _tau * (_targetTemperature / _temperature - 1.0));

    for (auto &molecule : simulationBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            auto velocities = molecule.getAtomVelocity(i);

            velocities *= berendsenFactor;

            molecule.setAtomVelocity(i, velocities);
        }
    }

    calculateTemperature(simulationBox, physicalData);
}