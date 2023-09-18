#include "velocityRescalingThermostat.hpp"

#include "molecule.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"

using thermostat::VelocityRescalingThermostat;

/**
 * @brief Copy constructor for Velocity Rescaling Thermostat
 *
 * @param other
 */
VelocityRescalingThermostat::VelocityRescalingThermostat(const VelocityRescalingThermostat &other)
    : Thermostat(other), _tau(other._tau){};

/**
 * @brief apply thermostat - Velocity Rescaling
 *
 * @link https://doi.org/10.1063/1.2408420
 *
 * @param simulationBox
 * @param physicalData
 */
void VelocityRescalingThermostat::applyThermostat(simulationBox::SimulationBox &simulationBox,
                                                  physicalData::PhysicalData   &physicalData)
{
    physicalData.calculateTemperature(simulationBox);

    _temperature = physicalData.getTemperature();

    const auto timeStep = settings::TimingsSettings::getTimeStep();

    const auto rescalingFactor =
        2.0 * ::sqrt(timeStep * _targetTemperature / (_temperature * double(simulationBox.getDegreesOfFreedom()) * _tau)) *
        std::normal_distribution<double>(0.0, 1.0)(_generator);

    const auto berendsenFactor = ::sqrt(1.0 + timeStep / _tau * (_targetTemperature / _temperature - 1.0) + rescalingFactor);

    for (const auto &atom : simulationBox.getAtoms())
        atom->scaleVelocity(berendsenFactor);

    physicalData.setTemperature(_temperature * berendsenFactor * berendsenFactor);
}