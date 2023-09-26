#include "berendsenThermostat.hpp"

#include "atom.hpp"              // for Atom
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings

#include <cmath>    // for sqrt
#include <memory>   // for __shared_ptr_access, shared_ptr
#include <vector>   // for vector

using thermostat::BerendsenThermostat;

/**
 * @brief apply thermostat - Berendsen
 *
 * @link https://doi.org/10.1063/1.448118
 *
 * @param simulationBox
 * @param physicalData
 */
void BerendsenThermostat::applyThermostat(simulationBox::SimulationBox &simulationBox, physicalData::PhysicalData &physicalData)
{
    physicalData.calculateTemperature(simulationBox);

    _temperature = physicalData.getTemperature();

    const auto berendsenFactor =
        ::sqrt(1.0 + settings::TimingsSettings::getTimeStep() / _tau * (_targetTemperature / _temperature - 1.0));

    for (const auto &atom : simulationBox.getAtoms())
        atom->scaleVelocity(berendsenFactor);

    physicalData.setTemperature(_temperature * berendsenFactor * berendsenFactor);
}