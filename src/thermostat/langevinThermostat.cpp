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

#include <algorithm>
#include <cmath>

#include "constants/conversionFactors.hpp"
#include "constants/natureConstants.hpp"
#include "globalTimer.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

namespace thermostat
{

    /**
     * @brief Constructor for Langevin Thermostat
     *
     * @details automatically calculates sigma from friction and target
     * temperature
     *
     * @param friction
     */
    LangevinThermostat::LangevinThermostat(double friction)
        : _friction(friction)
    {
    }

    /**
     * @brief Copy constructor for Langevin Thermostat
     *
     * @param other
     */
    LangevinThermostat::LangevinThermostat(const LangevinThermostat &other)
        : Thermostat(other), _friction(other._friction)
    {
    }

    /**
     * @brief Copy assignment operator for Langevin Thermostat
     *
     * @param other
     * @return LangevinThermostat&
     */
    LangevinThermostat &LangevinThermostat::operator=(
        const LangevinThermostat &other
    )
    {
        if (this != &other)
        {
            Thermostat::operator=(other);
            _friction = other._friction;
        }
        return *this;
    }

    /**
     * @brief Calculate sigma for Langevin Thermostat
     *
     * @param friction
     * @param targetTemperature
     */
    double LangevinThermostat::calculateSigma(
        double             friction,
        const Temperature &targetTemperature
    )
    {
        const auto unitConversion   = M2_TO_ANGSTROM2 * KG_TO_GRAM / FS_TO_S;
        const auto conversionFactor = UNIVERSAL_GAS_CONSTANT * unitConversion;

        const auto timeStep = settings::TimingsSettings::getTimeStep();
        const auto force =
            4.0 * friction * conversionFactor *
            targetTemperature.quantity_from_zero().numerical_value_in(K);

        return std::sqrt(force / timeStep.numerical_value_in(fs));
    }

    /**
     * @brief apply Langevin thermostat
     *
     * @details calculates the friction and random factor for each atom and
     * applies the Langevin thermostat to the velocities
     *
     * @param simulationBox
     */
    void LangevinThermostat::applyLangevin(
        molsys::SimulationBox &simulationBox,
        const Temperature     &targetTemperature
    )
    {
        const auto sigma = calculateSigma(_friction, targetTemperature);

        auto applyFriction = [this, sigma](auto &atom)
        {
            const auto mass     = atom->getMass();
            const auto timeStep = settings::TimingsSettings::getTimeStep();

            const auto propagationFactor =
                0.5 * timeStep.numerical_value_in(s) / mass;

            const linalg::Vec3D randomFactor = {
                _randomNumberGenerator.getNormalDistribution(0.0, 1.0),
                _randomNumberGenerator.getNormalDistribution(0.0, 1.0),
                _randomNumberGenerator.getNormalDistribution(0.0, 1.0)
            };

            const auto velocity = atom->getVelocity();
            auto       deltaVelocity =
                -propagationFactor * _friction * mass * velocity;

            deltaVelocity +=
                propagationFactor * sigma * std::sqrt(mass) * randomFactor;

            atom->addVelocity(deltaVelocity);
        };

        std::ranges::for_each(simulationBox.getAtoms(), applyFriction);
    }

    /**
     * @brief apply thermostat - Langevin
     *
     * @param simulationBox
     * @param physicalData
     */
    void LangevinThermostat::applyThermostat(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const Temperature          &targetTemperature
    )
    {
        auto _ =
            scopedTimer(TimerId::Thermostat, "LangevinThermostat - Full Step");

        applyLangevin(simulationBox, targetTemperature);
        physicalData.calculateTemperature(simulationBox);
    }

    /**
     * @brief apply thermostat half step - Langevin
     *
     * @note no temperature calculation
     *
     * @param simulationBox
     */
    void LangevinThermostat::applyThermostatHalfStep(
        molsys::SimulationBox &simulationBox,
        physicalData::PhysicalData & /*physicalData*/,
        const Temperature &targetTemperature
    )
    {
        auto _ =
            scopedTimer(TimerId::Thermostat, "LangevinThermostat - Half Step");

        applyLangevin(simulationBox, targetTemperature);
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the ThermostatType
     *
     * @return ThermostatType
     */
    ThermostatType LangevinThermostat::getThermostatType() const
    {
        return ThermostatType::LANGEVIN;
    }

}   // namespace thermostat
