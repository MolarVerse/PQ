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
    const auto conversionFactor = constants::_UNIVERSAL_GAS_CONSTANT_ * constants::_METER_SQUARED_TO_ANGSTROM_SQUARED_ *
                                  constants::_KG_TO_GRAM_ / constants::_FS_TO_S_;

    _sigma = std::sqrt(4.0 * friction * conversionFactor * targetTemperature / settings::TimingsSettings::getTimeStep());
}

/**
 * @brief Copy constructor for Langevin Thermostat
 *
 * @param other
 */
LangevinThermostat::LangevinThermostat(const LangevinThermostat &other)
    : Thermostat(other), _friction(other._friction), _sigma(other._sigma){};

/**
 * @brief apply Langevin thermostat
 *
 * @details calculates the friction and random factor for each atom and applies the Langevin thermostat to the velocities
 *
 * @param simBox
 */
void LangevinThermostat::applyLangevin(simulationBox::SimulationBox &simBox)
{
    auto applyFriction = [this](auto &atom)
    {
        const auto mass = atom->getMass();

        const auto propagationFactor            = 0.5 * settings::TimingsSettings::getTimeStep() * constants::_FS_TO_S_ / mass;
        const linearAlgebra::Vec3D randomFactor = {std::normal_distribution<double>(0.0, 1.0)(_generator),
                                                   std::normal_distribution<double>(0.0, 1.0)(_generator),
                                                   std::normal_distribution<double>(0.0, 1.0)(_generator)};

        auto dVelocity  = -propagationFactor * _friction * mass * atom->getVelocity();
        dVelocity      += propagationFactor * _sigma * std::sqrt(mass) * randomFactor;

        atom->addVelocity(dVelocity);
    };

    std::ranges::for_each(simBox.getAtoms(), applyFriction);
}

/**
 * @brief apply thermostat - Langevin
 *
 * @param simBox
 * @param data
 */
void LangevinThermostat::applyThermostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &data)
{
    applyLangevin(simBox);
    data.calculateTemperature(simBox);
}

/**
 * @brief apply thermostat half step - Langevin
 *
 * @note no temperature calculation
 *
 * @param simBox
 * @param data
 */
void LangevinThermostat::applyThermostatHalfStep(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &)
{
    applyLangevin(simBox);
}