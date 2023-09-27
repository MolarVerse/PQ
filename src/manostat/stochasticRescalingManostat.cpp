/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "stochasticRescalingManostat.hpp"

#include "constants/conversionFactors.hpp"           // for _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_
#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "exceptions.hpp"                            // for ExceptionType
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox
#include "thermostatSettings.hpp"                    // for ThermostatSettings
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "vector3d.hpp"                              // for Vec3D, operator/

#include <algorithm>    // for __for_each_fn
#include <functional>   // for identity
#include <math.h>       // for exp, pow, sqrt

using manostat::StochasticRescalingManostat;

/**
 * @brief copy constructor
 *
 * @param other
 */
StochasticRescalingManostat::StochasticRescalingManostat(const StochasticRescalingManostat &other)
    : Manostat(other), _tau(other._tau), _compressibility(other._compressibility){};

/**
 * @brief apply Stochastic Rescaling manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void StochasticRescalingManostat::applyManostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData)
{
    calculatePressure(simBox, physicalData);

    const auto compressibilityFactor = _compressibility * settings::TimingsSettings::getTimeStep() / _tau;

    const auto stochasticFactor =
        ::sqrt(2.0 * constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * settings::ThermostatSettings::getTargetTemperature() *
               compressibilityFactor / (simBox.getVolume()) * constants::_PRESSURE_FACTOR_) *
        std::normal_distribution<double>(0.0, 1.0)(_generator);

    const auto linearScalingFactor =
        ::pow(exp(-compressibilityFactor * (_targetPressure - _pressure) + stochasticFactor), 1.0 / 3.0);

    const auto scalingFactors = linearAlgebra::Vec3D(linearScalingFactor);

    simBox.scaleBox(scalingFactors);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulombRadiusCutOff(customException::ExceptionType::MANOSTATEXCEPTION);

    auto scalePositions  = [&scalingFactors](auto &molecule) { molecule.scale(scalingFactors); };
    auto scaleVelocities = [&scalingFactors](auto &atom) { atom->scaleVelocity(1.0 / scalingFactors); };

    std::ranges::for_each(simBox.getMolecules(), scalePositions);
    std::ranges::for_each(simBox.getAtoms(), scaleVelocities);
}