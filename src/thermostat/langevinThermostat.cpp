#include "langevinThermostat.hpp"

#include "constants.hpp"         // for constants::_BOLTZMANN_CONSTANT_
#include "physicalData.hpp"      // for physicalData::PhysicalData
#include "simulationBox.hpp"     // for simulationBox::SimulationBox
#include "timingsSettings.hpp"   // for settings::TimingsSettings::getTimeStep

#include <algorithm>   // for std::ranges::for_each
#include <cmath>       // for std::sqrt

using thermostat::LangevinThermostat;

/**
 * @brief Constructor for Langevin Thermostat
 *
 * @details automatically calculates sigma from friction and target temperature
 *
 * @param targetTemperature
 * @param friction
 */
LangevinThermostat::LangevinThermostat(const double targetTemperature, const double friction)
    : Thermostat(targetTemperature), _friction(friction)
{
    const auto conversionFactor =
        constants::_BOLTZMANN_CONSTANT_ * constants::_METER_SQUARED_TO_ANGSTROM_SQUARED_ * constants::_KG_TO_GRAM_;

    _sigma = std::sqrt(4.0 * friction * conversionFactor * targetTemperature / settings::TimingsSettings::getTimeStep());
}

void LangevinThermostat::applyLangevin(simulationBox::SimulationBox &simBox)
{
    auto applyFriction = [this](auto &atom)
    {
        const auto mass = atom->getMass();

        const auto                 propagationFactor = 0.5 * settings::TimingsSettings::getTimeStep() / mass;
        const linearAlgebra::Vec3D randomFactor      = {std::normal_distribution<double>(0.0, 1.0)(_generator),
                                                        std::normal_distribution<double>(0.0, 1.0)(_generator),
                                                        std::normal_distribution<double>(0.0, 1.0)(_generator)};

        auto dVelocity  = -propagationFactor * _friction * mass * atom->getVelocity();
        dVelocity      += propagationFactor * _sigma * std::sqrt(mass) * randomFactor;

        atom->addVelocity(dVelocity);
    };

    std::ranges::for_each(simBox.getAtoms(), applyFriction);
}

void LangevinThermostat::applyThermostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &data)
{
    applyLangevin(simBox);
    data.calculateTemperature(simBox);
}

void LangevinThermostat::applyThermostatHalfStep(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &)
{
    applyLangevin(simBox);
}