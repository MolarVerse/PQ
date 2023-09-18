#include "berendsenThermostat.hpp"

#include "molecule.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"

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