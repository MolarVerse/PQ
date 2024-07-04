/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "langevinThermostat.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <functional>   // for identity

#include "constants/conversionFactors.hpp"   // for _FS_TO_S_, _KG_TO_GRAM_
#include "constants/natureConstants.hpp"     // for _UNIVERSAL_GAS_CONSTANT_
#include "physicalData.hpp"                  // for PhysicalData
#include "simulationBox.hpp"                 // for SimulationBox
#include "thermostatSettings.hpp"            // for ThermostatType
#include "timingsSettings.hpp"               // for TimingsSettings
#include "vector3d.hpp"                      // for operator*, Vec3D

using thermostat::LangevinThermostat;
using namespace constants;
using namespace physicalData;
using namespace simulationBox;
using namespace settings;
using namespace linearAlgebra;

/**
 * @brief Constructor for Langevin Thermostat
 *
 * @details automatically calculates sigma from friction and target temperature
 *
 * @param targetTemperature
 * @param friction
 */
LangevinThermostat::LangevinThermostat(
    const double targetTemperature,
    const double friction
)
    : Thermostat(targetTemperature), _friction(friction)
{
    calculateSigma(friction, targetTemperature);
}

/**
 * @brief Copy constructor for Langevin Thermostat
 *
 * @param other
 */
LangevinThermostat::LangevinThermostat(const LangevinThermostat &other)
    : Thermostat(other), _friction(other._friction), _sigma(other._sigma)
{
}

/**
 * @brief Calculate sigma for Langevin Thermostat
 *
 * @param friction
 * @param targetTemperature
 */
void LangevinThermostat::calculateSigma(
    const double friction,
    const double targetTemperature
)
{
    const auto unitConversion   = _M2_TO_ANGSTROM2_ * _KG_TO_GRAM_ / _FS_TO_S_;
    const auto conversionFactor = _UNIVERSAL_GAS_CONSTANT_ * unitConversion;

    const auto timeStep = TimingsSettings::getTimeStep();
    const auto force    = 4.0 * friction * conversionFactor * targetTemperature;

    _sigma = std::sqrt(force / timeStep);
}

/**
 * @brief apply Langevin thermostat
 *
 * @details calculates the friction and random factor for each atom and applies
 * the Langevin thermostat to the velocities
 *
 * @param simBox
 */
void LangevinThermostat::applyLangevin(SimulationBox &simBox)
{
    auto applyFriction = [this](auto &atom)
    {
        const auto mass     = atom->getMass();
        const auto timeStep = TimingsSettings::getTimeStep();

        const auto propagationFactor = 0.5 * timeStep * _FS_TO_S_ / mass;

        const Vec3D randomFactor = {
            std::normal_distribution<double>(0.0, 1.0)(_generator),
            std::normal_distribution<double>(0.0, 1.0)(_generator),
            std::normal_distribution<double>(0.0, 1.0)(_generator)
        };

        const auto velocity = atom->getVelocity();
        auto       dv       = -propagationFactor * _friction * mass * velocity;

        dv += propagationFactor * _sigma * std::sqrt(mass) * randomFactor;

        atom->addVelocity(dv);
    };

    std::ranges::for_each(simBox.getAtoms(), applyFriction);
}

/**
 * @brief apply thermostat - Langevin
 *
 * @param simBox
 * @param data
 */
void LangevinThermostat::applyThermostat(
    SimulationBox &simBox,
    PhysicalData  &data
)
{
    startTimingsSection("LangevinThermostat - Full Step");

    applyLangevin(simBox);
    data.calculateTemperature(simBox);

    stopTimingsSection("LangevinThermostat - Full Step");
}

/**
 * @brief apply thermostat half step - Langevin
 *
 * @note no temperature calculation
 *
 * @param simBox
 * @param data
 */
void LangevinThermostat::
    applyThermostatHalfStep(SimulationBox &simBox, PhysicalData &)
{
    startTimingsSection("LangevinThermostat - Half Step");

    applyLangevin(simBox);

    stopTimingsSection("LangevinThermostat - Half Step");
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief Set target temperature for Langevin Thermostat and calculate sigma
 *
 * @param targetTemperature
 */
void LangevinThermostat::setTargetTemperature(const double targetTemperature)
{
    _targetTemperature = targetTemperature;
    calculateSigma(_friction, targetTemperature);
}

/**
 * @brief Set the friction for Langevin Thermostat
 *
 * @param friction
 */
void LangevinThermostat::setFriction(const double friction)
{
    _friction = friction;
}

/**
 * @brief Set the sigma for Langevin Thermostat
 *
 * @param sigma
 */
void LangevinThermostat::setSigma(const double sigma) { _sigma = sigma; }

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief Get the friction for Langevin Thermostat
 *
 * @return double
 */
double LangevinThermostat::getFriction() const { return _friction; }

/**
 * @brief Get the sigma for Langevin Thermostat
 *
 * @return double
 */
double LangevinThermostat::getSigma() const { return _sigma; }

/**
 * @brief get the ThermostatType
 *
 * @return ThermostatType
 */
settings::ThermostatType LangevinThermostat::getThermostatType() const
{
    return settings::ThermostatType::LANGEVIN;
}