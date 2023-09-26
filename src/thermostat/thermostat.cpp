#include "thermostat.hpp"

#include "physicalData.hpp"

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

using thermostat::Thermostat;

/**
 * @brief apply thermostat - base class
 *
 * @note here base class represents none thermostat
 *
 * @param simulationBox
 * @param physicalData
 */
void Thermostat::applyThermostat(simulationBox::SimulationBox &simulationBox, physicalData::PhysicalData &physicalData)
{
    physicalData.calculateTemperature(simulationBox);
}