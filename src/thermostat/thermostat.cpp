#include "thermostat.hpp"
#include "constants.hpp"

#include <cmath>

using namespace std;

void Thermostat::calculateTemperature(const SimulationBox &simulationBox, PhysicalData &physicalData)
{
    auto temperature = 0.0;
    auto velocities = Vec3D();

    for (const auto &molecule : simulationBox._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            velocities = molecule.getAtomVelocities(i);

            const auto mass = molecule.getMass(i);

            temperature += mass * (velocities[0] * velocities[0] + velocities[1] * velocities[1] + velocities[2] * velocities[2]);
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

    auto velocities = Vec3D();
    const auto berendsenFactor = sqrt(1.0 + _timestep / _tau * (_targetTemperature / _temperature - 1.0));

    for (auto &molecule : simulationBox._molecules)
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            velocities = molecule.getAtomVelocities(i);

            velocities[0] *= berendsenFactor;
            velocities[1] *= berendsenFactor;
            velocities[2] *= berendsenFactor;

            molecule.setAtomVelocities(i, velocities);
        }
    }

    calculateTemperature(simulationBox, physicalData);
}