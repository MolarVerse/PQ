#include "thermostat.hpp"
#include "constants.hpp"

using namespace std;

void Thermostat::calculateTemperature(const SimulationBox &simulationBox, OutputData &outputData)
{
    auto temperature = 0.0;
    auto velocities = vector<double>(3);

    for (const auto &molecule : simulationBox._molecules)
    {
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            molecule.getAtomVelocities(i, velocities);

            auto mass = molecule.getMass(i);

            temperature += mass * (velocities[0] * velocities[0] + velocities[1] * velocities[1] + velocities[2] * velocities[2]);
        }
    }

    temperature *= _TEMPERATURE_FACTOR_ / simulationBox.getDegreesOfFreedom();

    setTemperature(temperature);
    outputData.addAverageTemperature(temperature);
}