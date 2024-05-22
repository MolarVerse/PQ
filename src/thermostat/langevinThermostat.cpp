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
#include "timingsSettings.hpp"               // for TimingsSettings
#include "vector3d.hpp"                      // for operator*, Vec3D

using thermostat::LangevinThermostat;

/**
 * @brief Constructor for Langevin Thermostat
 *
 * @details automatically calculates sigma from friction and target temperature
 *
 * @param targetTemp
 * @param friction
 */
LangevinThermostat::LangevinThermostat(
    const double targetTemp,
    const double friction
)
    : Thermostat(targetTemp), _friction(friction)
{
    const auto R        = constants::_UNIVERSAL_GAS_CONSTANT_;
    const auto m2_to_A2 = constants::_METER_SQUARED_TO_ANGSTROM_SQUARED_;
    const auto kg_to_g  = constants::_KG_TO_GRAM_;
    const auto fs_to_s  = constants::_FS_TO_S_;
    const auto dt       = settings::TimingsSettings::getTimeStep();

    const auto conversionFactor = R * m2_to_A2 * kg_to_g / fs_to_s;

    _sigma = std::sqrt(4.0 * friction * conversionFactor * targetTemp / dt);
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
 * @details calculates the friction and random factor for each atom and applies
 * the Langevin thermostat to the velocities
 *
 * @param simBox
 */
void LangevinThermostat::applyLangevin(simulationBox::SimulationBox &simBox)
{
    auto applyFriction = [this](auto &atom)
    {
        const auto dt   = settings::TimingsSettings::getTimeStep();
        const auto mass = atom->getMass();

        const auto propFactor = 0.5 * dt * constants::_FS_TO_S_ / mass;

        const linearAlgebra::Vec3D randomFactor = {
            std::normal_distribution<double>(0.0, 1.0)(_generator),
            std::normal_distribution<double>(0.0, 1.0)(_generator),
            std::normal_distribution<double>(0.0, 1.0)(_generator)
        };

        auto dVelocity  = -propFactor * _friction * mass * atom->getVelocity();
        dVelocity      += propFactor * _sigma * std::sqrt(mass) * randomFactor;

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
void LangevinThermostat::applyThermostat(
    simulationBox::SimulationBox &simBox,
    physicalData::PhysicalData   &data
)
{
    startTimingsSection("Langevin - second half step");

    applyLangevin(simBox);
    data.calculateTemperature(simBox);

    stopTimingsSection("Langevin - second half step");
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
    applyThermostatHalfStep(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &)
{
    startTimingsSection("Langevin - first half step");

    applyLangevin(simBox);

    stopTimingsSection("Langevin - first half step");
}