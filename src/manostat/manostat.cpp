#include "manostat.hpp"

#include "constants.hpp"         // for _PRESSURE_FACTOR_
#include "exceptions.hpp"        // for ExceptionType
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for pow
#include <functional>   // for identity, function

using namespace manostat;

/**
 * @brief calculate the pressure of the system
 *
 * @param physicalData
 */
void Manostat::calculatePressure(const simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    const auto ekinVirial  = physicalData.getKineticEnergyVirialVector();
    const auto forceVirial = physicalData.getVirial();
    const auto volume      = box.getVolume();

    _pressureVector = (2.0 * ekinVirial + forceVirial) / volume * constants::_PRESSURE_FACTOR_;

    _pressure = mean(_pressureVector);

    physicalData.setPressure(_pressure);
}

/**
 * @brief apply dummy manostat for NVT ensemble
 *
 * @param physicalData
 */
void Manostat::applyManostat(simulationBox::SimulationBox &box, physicalData::PhysicalData &physicalData)
{
    calculatePressure(box, physicalData);
}