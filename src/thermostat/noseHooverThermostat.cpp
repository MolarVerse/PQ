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

#include "noseHooverThermostat.hpp"

#include <algorithm>
#include <cstddef>

#include "constants/conversionFactors.hpp"
#include "constants/internalConversionFactors.hpp"
#include "exceptions.hpp"
#include "globalTimer.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "timingsSettings.hpp"
#include "units.hpp"

namespace thermostat
{

    /**
     * @brief Construct a new Nose Hoover Thermostat:: Nose Hoover Thermostat
     * object
     *
     * @param nhSettings Reference to the NoseHooverSettings object containing
     * the Nose-Hoover chain parameters such as chi, zeta, and coupling
     * frequency.
     */
    NoseHooverThermostat::NoseHooverThermostat(
        const NoseHooverSettings &nhSettings
    )
        : _nhSettings(nhSettings)
    {
        const auto length = nhSettings.getNoseHooverChainLength() + 1;
        _chi.reserve(length);
        _zeta.reserve(length);

        _chi.resize(length);
        _zeta.resize(length);

        if (nhSettings.isChiAndZetaSet())
        {
            for (const auto &[idx, chi] : nhSettings.getChi()) _chi[idx] = chi;

            for (const auto &[idx, zeta] : nhSettings.getZeta())
                _zeta[idx] = zeta;
        }
    }

    /**
     * @brief throw if the Nose-Hoover equations divide by zero
     *
     * @details they divide by the thermal energy of the target temperature
     * and by the degrees of freedom
     *
     * @param simulationBox
     *
     * @throw exc::UserInputException for a zero target temperature or zero
     * degrees of freedom
     */
    void NoseHooverThermostat::_requireCoupling(
        const molsys::SimulationBox &simulationBox,
        const Temperature           &targetTemperature
    )
    {
        if (targetTemperature.quantity_from_zero() <= 0.0)
        {
            throw exc::UserInputException(
                "Cannot apply the Nose-Hoover thermostat with a target "
                "temperature of zero or below."
            );
        }

        _requireDegreesOfFreedom(simulationBox, "the Nose-Hoover thermostat");
    }

    /**
     * @brief applies the Nose-Hoover thermostat on the forces
     *
     * @details the Nose-Hoover thermostat is applied on the forces of the atoms
     * after force calculation
     *
     * @param simulationBox simulation box
     */
    void NoseHooverThermostat::applyThermostatOnForces(
        molsys::SimulationBox &simulationBox,
        const Temperature     &targetTemperature
    )
    {
        auto _ = scopedTimer(TimerId::Thermostat, "Nose-Hoover - Forces");

        _requireCoupling(simulationBox, targetTemperature);

        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;
        const auto kT_target =
            boltzmannConstant *
            targetTemperature.quantity_from_zero().numerical_value_in(K);

        const auto degreesOfFreedom =
            static_cast<double>(simulationBox.getDegreesOfFreedom());
        const auto couplingFreq = _nhSettings->getNoseHooverCouplingFrequency();
        const auto couplingFreqSquared =
            (couplingFreq * couplingFreq).numerical_value_in(Hz * Hz);

        auto factor  = _chi[0] * couplingFreqSquared;
        factor      /= (kT_target * degreesOfFreedom);
        factor      *= MOMENTUM_TO_FORCE;

        auto applyNoseHoover = [factor](auto &atom)
        { atom->addForce(-factor * atom->getVelocity() * atom->getMass()); };

        std::ranges::for_each(simulationBox.getAtoms(), applyNoseHoover);
    }

    /**
     * @brief applies the Nose-Hoover thermostat on the velocities
     *
     * @details the Nose-Hoover thermostat is applied on the velocities of the
     * atoms after velocity integration
     *
     * @param simulationBox simulation box
     * @param physicalData physical data
     */
    void NoseHooverThermostat::applyThermostat(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const Temperature          &targetTemperature
    )
    {
        auto _ = scopedTimer(TimerId::Thermostat, "Nose-Hoover - Velocities");

        _requireCoupling(simulationBox, targetTemperature);

        physicalData.calculateTemperature(simulationBox);

        setTemperature(physicalData.getTemperature());

        const auto degreesOfFreedom =
            static_cast<double>(simulationBox.getDegreesOfFreedom());

        const auto couplingFreq = _nhSettings->getNoseHooverCouplingFrequency();
        const auto couplingFreqSquared =
            (couplingFreq * couplingFreq).numerical_value_in(Hz * Hz);
        const auto timestep =
            settings::TimingsSettings::getTimeStep().numerical_value_in(s);
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            getTemperature().quantity_from_zero().numerical_value_in(K);
        const auto thermalEnergyTarget =
            boltzmannConstant *
            targetTemperature.quantity_from_zero().numerical_value_in(K);

        auto chi = (thermalEnergy - thermalEnergyTarget) * degreesOfFreedom;
        chi -= _chi[0] * _chi[1] / thermalEnergyTarget * couplingFreqSquared;

        _chi[0] += timestep * chi;

        auto helpRatio = _chi[0] / (thermalEnergyTarget * degreesOfFreedom) *
                         couplingFreqSquared;

        _zeta[0]   += helpRatio * timestep;
        auto ratio  = helpRatio * _chi[0];

        auto energyMomentum = ratio;
        auto energyFriction = degreesOfFreedom * _zeta[0];

        for (size_t i = 1; i < _chi.size() - 1; ++i)
        {
            chi  = ratio;
            chi -= thermalEnergyTarget;
            chi -= _chi[i] * _chi[i + 1] / thermalEnergyTarget *
                   couplingFreqSquared;

            _chi[i] += timestep * chi;

            helpRatio  = _chi[i] / thermalEnergyTarget * couplingFreqSquared;
            _zeta[i]  += helpRatio * timestep;
            ratio      = helpRatio * _chi[i];

            energyMomentum += ratio;
            energyFriction += _zeta[i];
        }

        physicalData.setNoseHooverMomentumEnergy(energyMomentum);
        physicalData.setNoseHooverFrictionEnergy(energyFriction);
    }
    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the chi values of the Nose-Hoover thermostat
     *
     * @details only the real chain links are returned, not the internal zero
     * boundary term
     *
     * @return std::vector<double>
     */
    std::vector<double> NoseHooverThermostat::getChi() const
    {
        return {_chi.begin(), _chi.end() - 1};
    }

    /**
     * @brief get the zeta values of the Nose-Hoover thermostat
     *
     * @details only the real chain links are returned, not the internal
     * boundary term
     *
     * @return std::vector<double>
     */
    std::vector<double> NoseHooverThermostat::getZeta() const
    {
        return {_zeta.begin(), _zeta.end() - 1};
    }

    /**
     * @brief get the ThermostatType
     *
     * @return ThermostatType
     */
    ThermostatType NoseHooverThermostat::getThermostatType() const
    {
        return ThermostatType::NOSE_HOOVER;
    }

}   // namespace thermostat
