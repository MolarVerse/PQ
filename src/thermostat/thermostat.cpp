#include "thermostat.hpp"

#include "constants.hpp"

#include <cmath>

using namespace std;
using namespace simulationBox;
using namespace thermostat;
using namespace physicalData;
using namespace config;

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
    physicalData.calculateTemperature(simulationBox);
}

/**
 * @brief apply thermostat - Berendsen
 *
 * @param simulationBox
 * @param physicalData
 */
void BerendsenThermostat::applyThermostat(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    physicalData.calculateTemperature(simulationBox);

    _temperature = physicalData.getTemperature();

    const auto berendsenFactor = sqrt(1.0 + _timestep / _tau * (_targetTemperature / _temperature - 1.0));

    for (auto &molecule : simulationBox.getMolecules())
        molecule.scaleVelocities(berendsenFactor);

    physicalData.setTemperature(_temperature * berendsenFactor * berendsenFactor);
}