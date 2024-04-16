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

#include "stochasticRescalingManostat.hpp"

#include "constants/conversionFactors.hpp"           // for _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_
#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "exceptions.hpp"                            // for ExceptionType
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox
#include "staticMatrix3x3.hpp"                       // for diagonal, diagonalMatrix
#include "thermostatSettings.hpp"                    // for ThermostatSettings
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "vector3d.hpp"                              // for Vec3D, operator/

#include <algorithm>    // for __for_each_fn
#include <cmath>        // for exp, pow, sqrt
#include <functional>   // for identity

using manostat::AnisotropicStochasticRescalingManostat;
using manostat::FullAnisotropicStochasticRescalingManostat;
using manostat::SemiIsotropicStochasticRescalingManostat;
using manostat::StochasticRescalingManostat;

using namespace linearAlgebra;

/**
 * @brief copy constructor for Stochastic Rescaling Manostat
 *
 * @param other
 */
StochasticRescalingManostat::StochasticRescalingManostat(const StochasticRescalingManostat &other)
    : Manostat(other), _tau(other._tau), _compressibility(other._compressibility), _dt(other._dt){};

/**
 * @brief Construct a new Stochastic Rescaling Manostat:: Stochastic Rescaling Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 */
StochasticRescalingManostat::StochasticRescalingManostat(const double targetPressure,
                                                         const double tau,
                                                         const double compressibility)
    : Manostat(targetPressure), _tau(tau), _compressibility(compressibility)
{
    _dt = settings::TimingsSettings::getTimeStep();
}

/**
 * @brief apply Stochastic Rescaling manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void StochasticRescalingManostat::applyManostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData)
{
    calculatePressure(simBox, physicalData);

    const auto mu = calculateMu(simBox.getVolume());

    simBox.scaleBox(mu);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulombRadiusCutOff(customException::ExceptionType::MANOSTATEXCEPTION);

    auto scalePositions  = [&mu, &simBox](auto &molecule) { molecule.scale(mu, simBox.getBox()); };
    auto scaleVelocities = [&mu, &simBox](auto &atom) { atom->scaleVelocityOrthogonalSpace(inverse(mu), simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), scalePositions);
    std::ranges::for_each(simBox.getAtoms(), scaleVelocities);
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat (isotropic)
 *
 * @param volume
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::tensor3D StochasticRescalingManostat::calculateMu(const double volume)
{
    const auto compressibilityFactor = _compressibility * _dt / _tau;
    const auto kT = constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * settings::ThermostatSettings::getTargetTemperature();
    const auto randomFactor = std::normal_distribution<double>(0.0, 1.0)(_generator);

    const auto stochasticFactor = ::sqrt(2.0 * kT * compressibilityFactor / volume * constants::_PRESSURE_FACTOR_) * randomFactor;

    return diagonalMatrix(::exp(-compressibilityFactor * (_targetPressure - _pressure) + stochasticFactor / 3.0));
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat (semi-isotropic)
 *
 * @param volume
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::tensor3D SemiIsotropicStochasticRescalingManostat::calculateMu(const double volume)
{
    const auto compressibilityFactor = _compressibility * _dt / _tau;
    const auto kT = constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * settings::ThermostatSettings::getTargetTemperature();
    const auto randomFactor = std::normal_distribution<double>(0.0, 1.0)(_generator);

    const auto stochasticFactor_xy =
        ::sqrt(4.0 / 3.0 * kT * compressibilityFactor / volume * constants::_PRESSURE_FACTOR_) * randomFactor;
    const auto stochasticFactor_z =
        ::sqrt(2.0 / 3.0 * kT * compressibilityFactor / volume * constants::_PRESSURE_FACTOR_) * randomFactor;

    const auto p_xyz = diagonal(_pressureTensor);
    const auto p_x   = p_xyz[_2DIsotropicAxes[0]];
    const auto p_y   = p_xyz[_2DIsotropicAxes[1]];
    const auto p_xy  = (p_x + p_y) / 2.0;
    const auto p_z   = p_xyz[_2DAnisotropicAxis];

    const auto mu_xy = ::exp(-compressibilityFactor * (_targetPressure - p_xy) / 3.0 + stochasticFactor_xy / 2.0);
    const auto mu_z  = ::exp(-compressibilityFactor * (_targetPressure - p_z) / 3.0 + stochasticFactor_z);

    linearAlgebra::Vec3D mu;

    mu[_2DIsotropicAxes[0]] = mu_xy;
    mu[_2DIsotropicAxes[1]] = mu_xy;
    mu[_2DAnisotropicAxis]  = mu_z;

    return diagonalMatrix(mu);
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat (anisotropic)
 *
 * @param volume
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::tensor3D AnisotropicStochasticRescalingManostat::calculateMu(const double volume)
{
    const auto compressibilityFactor = _compressibility * _dt / _tau;
    const auto kT = constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * settings::ThermostatSettings::getTargetTemperature();
    const auto randomFactor = std::normal_distribution<double>(0.0, 1.0)(_generator);

    const auto stochasticFactor =
        ::sqrt(2.0 / 3.0 * kT * compressibilityFactor / volume * constants::_PRESSURE_FACTOR_) * randomFactor;

    return diagonalMatrix(exp(-compressibilityFactor * (_targetPressure - diagonal(_pressureTensor)) / 3.0 + stochasticFactor));
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat (full anisotropic including angles)
 *
 * @param volume
 * @return linearAlgebra::tensor3D
 */
linearAlgebra::tensor3D FullAnisotropicStochasticRescalingManostat::calculateMu(const double volume)
{
    const auto compressibilityFactor = _compressibility * _dt / _tau;
    const auto kT = constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * settings::ThermostatSettings::getTargetTemperature();
    const auto randomFactor = std::normal_distribution<double>(0.0, 1.0)(_generator);

    const auto stochasticFactor =
        ::sqrt(2.0 / 3.0 * kT * compressibilityFactor / volume * constants::_PRESSURE_FACTOR_) * randomFactor;

    return exp(-compressibilityFactor * (diagonalMatrix(_targetPressure) - _pressureTensor) / 3.0 + stochasticFactor);
}