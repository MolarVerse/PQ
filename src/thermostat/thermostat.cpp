#include "thermostat.hpp"

#include "constants.hpp"

#include <cmath>

using namespace std;
using namespace simulationBox;
using namespace thermostat;

/**
 * @brief calculate temperature
 *
 * @param simulationBox
 * @param physicalData
 */
void Thermostat::calculateTemperature(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    _temperature = 0.0;

    for (const auto &molecule : simulationBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto velocities = molecule.getAtomVelocity(i);
            const auto mass       = molecule.getAtomMass(i);

            _temperature += mass * normSquared(velocities);
        }
    }

    _temperature *= _TEMPERATURE_FACTOR_ / simulationBox.getDegreesOfFreedom();

    physicalData.setTemperature(_temperature);
}

/**
 * @brief apply thermostat - base class
 *
 * @note here base class represents none thermostat
 *
 * @param simulationBox
 * @param physicalData
 */
void Thermostat::applyThermostat(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    calculateTemperature(simulationBox, physicalData);
}

/**
 * @brief apply thermostat - Berendsen
 *
 * @param simulationBox
 * @param physicalData
 */
void BerendsenThermostat::applyThermostat(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    calculateTemperature(simulationBox, physicalData);

    const auto berendsenFactor = sqrt(1.0 + _timestep / _tau * (_targetTemperature / _temperature - 1.0));

    for (auto &molecule : simulationBox.getMolecules())
        molecule.scaleVelocities(berendsenFactor);

    calculateTemperature(simulationBox, physicalData);
}