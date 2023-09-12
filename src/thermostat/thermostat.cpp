#include "thermostat.hpp"

#include "molecule.hpp"          // for Molecule
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings

#include <cmath>    // for sqrt
#include <vector>   // for vector

using namespace thermostat;

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

    std::ranges::for_each(simulationBox.getAtoms(), [berendsenFactor](auto &atom) { atom->scaleVelocity(berendsenFactor); });

    physicalData.setTemperature(_temperature * berendsenFactor * berendsenFactor);
}