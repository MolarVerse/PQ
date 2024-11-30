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

#include <algorithm>    // for __for_each_fn
#include <cstddef>      // for size_t
#include <functional>   // for identity

#include "constants/conversionFactors.hpp"   // for _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_, _FS_TO_S_
#include "constants/internalConversionFactors.hpp"   // for _MOMENTUM_TO_FORCE_
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox
#include "thermostatSettings.hpp"                    // for ThermostatType
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "vector3d.hpp"                              // for operator*

using thermostat::NoseHooverThermostat;
using namespace constants;
using namespace settings;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief Construct a new Nose Hoover Thermostat:: Nose Hoover Thermostat object
 *
 * @param targetTemp
 * @param chi
 * @param zeta
 * @param couplingFrequency
 */
NoseHooverThermostat::NoseHooverThermostat(
    const double               targetTemp,
    const std::vector<double> &chi,
    const std::vector<double> &zeta,
    const double               couplingFrequency
)
    : Thermostat(targetTemp),
      _chi(chi),
      _zeta(zeta),
      _couplingFrequency(couplingFrequency) {};

/**
 * @brief applies the Nose-Hoover thermostat on the velocities
 *
 * @details the Nose-Hoover thermostat is applied on the velocities of the atoms
 * after velocity integration
 *
 * @param simBox simulation box
 * @param physicalData physical data
 */
void NoseHooverThermostat::applyThermostat(
    SimulationBox &simBox,
    PhysicalData  &physicalData
)
{
    startTimingsSection("Nose-Hoover - Velocities");

    physicalData.calculateTemperature(simBox);

    _temperature = physicalData.getTemperature();

    const auto degreesOfFreedom    = double(simBox.getDegreesOfFreedom());
    const auto couplingFreqSquared = _couplingFrequency * _couplingFrequency;

    const auto dt = TimingsSettings::getTimeStep();
    const auto kB = _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_;

    const auto timestep  = dt * _FS_TO_S_;
    const auto kT        = kB * _temperature;
    const auto kT_target = kB * _targetTemperature;

    auto chi  = (kT - kT_target) * degreesOfFreedom;
    chi      -= _chi[0] * _chi[1] / kT_target * couplingFreqSquared;

    _chi[0] += timestep * chi;

    auto ratio = _chi[0] / (kT_target * degreesOfFreedom) * couplingFreqSquared;

    _zeta[0] += ratio * timestep;
    ratio    *= _chi[0];

    auto energyMomentum = ratio;
    auto energyFriction = degreesOfFreedom * _zeta[0];

    for (size_t i = 1; i < _chi.size() - 1; ++i)
    {
        chi  = ratio;
        chi -= kT_target;
        chi -= _chi[i] * _chi[i + 1] / kT_target * couplingFreqSquared;

        _chi[i] += timestep * chi;

        ratio     = _chi[i] / kT_target * couplingFreqSquared;
        _zeta[i] += ratio * timestep;
        ratio    *= _chi[i];

        energyMomentum += ratio;
        energyFriction += _zeta[i];
    }

    physicalData.setNoseHooverMomentumEnergy(energyMomentum);
    physicalData.setNoseHooverFrictionEnergy(energyFriction);

    stopTimingsSection("Nose-Hoover - Velocities");
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the chi values of the Nose-Hoover thermostat
 *
 * @return std::vector<double>
 */
std::vector<double> NoseHooverThermostat::getChi() const { return _chi; }

/**
 * @brief get the zeta values of the Nose-Hoover thermostat
 *
 * @return std::vector<double>
 */
std::vector<double> NoseHooverThermostat::getZeta() const { return _zeta; }

/**
 * @brief get the coupling frequency of the Nose-Hoover thermostat
 *
 * @return double
 */
double NoseHooverThermostat::getCouplingFrequency() const
{
    return _couplingFrequency;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the chi value at the given index
 *
 * @param index
 * @param chi
 */
void NoseHooverThermostat::setChi(const unsigned int index, const double chi)
{
    _chi[index] = chi;
}

/**
 * @brief set the chi values of the Nose-Hoover thermostat
 *
 * @param chi
 */
void NoseHooverThermostat::setChi(const std::vector<double> &chi)
{
    _chi = chi;
}

/**
 * @brief set the zeta value at the given index
 *
 * @param index
 * @param zeta
 */
void NoseHooverThermostat::setZeta(const unsigned int index, const double zeta)
{
    _zeta[index] = zeta;
}

/**
 * @brief set the zeta values of the Nose-Hoover thermostat
 *
 * @param zeta
 */
void NoseHooverThermostat::setZeta(const std::vector<double> &zeta)
{
    _zeta = zeta;
}

/**
 * @brief set the coupling frequency of the Nose-Hoover thermostat
 *
 * @param couplingFrequency
 */
void NoseHooverThermostat::setCouplingFrequency(const double couplingFrequency)
{
    _couplingFrequency = couplingFrequency;
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