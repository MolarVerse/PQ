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

#include <algorithm>    // for __for_each_fn
#include <cmath>        // for exp, pow, sqrt
#include <functional>   // for identity

#include "constants/conversionFactors.hpp"   // for _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_
#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "exceptions.hpp"                            // for ExceptionType
#include "physicalData.hpp"                          // for PhysicalData
#include "simulationBox.hpp"                         // for SimulationBox
#include "staticMatrix.hpp"         // for diagonal, diagonalMatrix
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "timingsSettings.hpp"      // for TimingsSettings
#include "vector3d.hpp"             // for Vec3D, operator/

using namespace linearAlgebra;
using namespace manostat;
using namespace settings;
using namespace simulationBox;
using namespace physicalData;
using namespace customException;
using namespace constants;
using namespace linearAlgebra;

/**
 * @brief copy constructor for Stochastic Rescaling Manostat
 *
 * @param other
 */
StochasticRescalingManostat::StochasticRescalingManostat(
    const StochasticRescalingManostat &other
)
    : Manostat(other),
      _tau(other._tau),
      _compressibility(other._compressibility),
      _dt(other._dt) {};

/**
 * @brief Construct a new Stochastic Rescaling Manostat:: Stochastic Rescaling
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 * @param anisotropicAxis
 * @param isotropicAxes
 * @return SemiIsotropicStochasticRescalingManostat::
 */
SemiIsotropicStochasticRescalingManostat::
    SemiIsotropicStochasticRescalingManostat(
        const double               targetPressure,
        const double               tau,
        const double               compressibility,
        const size_t               anisotropicAxis,
        const std::vector<size_t> &isotropicAxes
    )
    : StochasticRescalingManostat(targetPressure, tau, compressibility),
      _2DAnisotropicAxis(anisotropicAxis),
      _2DIsotropicAxes(isotropicAxes) {};

/**
 * @brief Construct a new Stochastic Rescaling Manostat:: Stochastic Rescaling
 * Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 */
StochasticRescalingManostat::StochasticRescalingManostat(
    const double targetPressure,
    const double tau,
    const double compressibility
)
    : Manostat(targetPressure), _tau(tau), _compressibility(compressibility)
{
    _dt = TimingsSettings::getTimeStep();
}

/**
 * @brief apply Stochastic Rescaling manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void StochasticRescalingManostat::applyManostat(
    simulationBox::SimulationBox &simBox,
    physicalData::PhysicalData   &physicalData
)
{
    startTimingsSection("Stochastic Rescaling");

    calculatePressure(simBox, physicalData);

    const auto mu = calculateMu(simBox.getVolume());

    simBox.scaleBox(mu);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulRadiusCutOff(ExceptionType::MANOSTATEXCEPTION);

    auto scalePositions = [&mu, &simBox](auto &molecule)
    { molecule.scale(mu, simBox.getBox()); };

    auto scaleVelocities = [&mu, &simBox](auto &atom)
    { atom->scaleVelocityOrthogonalSpace(inverse(mu), simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), scalePositions);
    std::ranges::for_each(simBox.getAtoms(), scaleVelocities);

    stopTimingsSection("Stochastic Rescaling");
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat
 * (isotropic)
 *
 * @param volume
 * @return Vec3D
 */
tensor3D StochasticRescalingManostat::calculateMu(const double volume)
{
    const auto compress = _compressibility * _dt / _tau;
    const auto kb       = _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = std::normal_distribution<double>(0.0, 1.0)(_generator);

    auto stochasticFactor  = 2.0 * kT * compress / volume;
    stochasticFactor      *= _PRESSURE_FACTOR_;
    stochasticFactor       = ::sqrt(stochasticFactor) * random;

    const auto deltaP = _targetPressure - _pressure;

    return diagonalMatrix(::exp(-compress * (deltaP) + stochasticFactor / 3.0));
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat
 * (semi-isotropic)
 *
 * @param volume
 * @return Vec3D
 */
tensor3D SemiIsotropicStochasticRescalingManostat::calculateMu(
    const double volume
)
{
    const auto compress = _compressibility * _dt / _tau;
    const auto kb       = _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = std::normal_distribution<double>(0.0, 1.0)(_generator);

    auto stochasticFactor  = 1 / 3.0 * kT * compress / volume;
    stochasticFactor      *= _PRESSURE_FACTOR_;

    const auto stochasticFactor_xy = ::sqrt(4.0 * stochasticFactor) * random;
    const auto stochasticFactor_z  = ::sqrt(2.0 * stochasticFactor) * random;

    const auto p_xyz = diagonal(_pressureTensor);
    const auto p_x   = p_xyz[_2DIsotropicAxes[0]];
    const auto p_y   = p_xyz[_2DIsotropicAxes[1]];
    const auto p_xy  = (p_x + p_y) / 2.0;
    const auto p_z   = p_xyz[_2DAnisotropicAxis];

    const auto deltaPxy = _targetPressure - p_xy;
    const auto deltaPz  = _targetPressure - p_z;

    // clang-format off
    const auto mu_xy = ::exp(-compress * deltaPxy / 3.0 + stochasticFactor_xy / 2.0);
    const auto mu_z  = ::exp(-compress * deltaPz / 3.0 + stochasticFactor_z);
    // clang-format on

    Vec3D mu;

    mu[_2DIsotropicAxes[0]] = mu_xy;
    mu[_2DIsotropicAxes[1]] = mu_xy;
    mu[_2DAnisotropicAxis]  = mu_z;

    return diagonalMatrix(mu);
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat
 * (anisotropic)
 *
 * @param volume
 * @return Vec3D
 */
tensor3D AnisotropicStochasticRescalingManostat::calculateMu(const double volume
)
{
    const auto compress = _compressibility * _dt / _tau;
    const auto kb       = _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = std::normal_distribution<double>(0.0, 1.0)(_generator);

    auto stochasticFactor  = 2.0 / 3.0 * kT * compress / volume;
    stochasticFactor      *= _PRESSURE_FACTOR_;
    stochasticFactor       = ::sqrt(stochasticFactor) * random;

    const auto deltaP = _targetPressure - diagonal(_pressureTensor);

    return diagonalMatrix(exp(-compress * (deltaP) / 3.0 + stochasticFactor));
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat (full
 * anisotropic including angles)
 *
 * @param volume
 * @return tensor3D
 */
tensor3D FullAnisotropicStochasticRescalingManostat::calculateMu(
    const double volume
)
{
    const auto compress = _compressibility * _dt / _tau;
    const auto kb       = _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = std::normal_distribution<double>(0.0, 1.0)(_generator);

    auto stochasticFactor  = 2.0 / 3.0 * kT * compress / volume;
    stochasticFactor      *= _PRESSURE_FACTOR_;
    stochasticFactor       = ::sqrt(stochasticFactor) * random;

    const auto deltaP = diagonalMatrix(_targetPressure) - _pressureTensor;
    auto       mu     = expPade(-compress * deltaP / 3.0 + stochasticFactor);

    // rotate mu to the original coordinate system
    // first-order approximation

    mu[0][1] += mu[1][0];
    mu[0][2] += mu[2][0];
    mu[1][2] += mu[2][1];

    mu[1][0] = 0.0;
    mu[2][0] = 0.0;
    mu[2][1] = 0.0;

    return mu;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get tau (relaxation time)
 *
 * @return double
 */
double StochasticRescalingManostat::getTau() const { return _tau; }

/**
 * @brief get compressibility
 *
 * @return double
 */
double StochasticRescalingManostat::getCompressibility() const
{
    return _compressibility;
}

/**
 * @brief get the manostat type
 *
 * @return ManostatType
 */
ManostatType StochasticRescalingManostat::getManostatType() const
{
    return ManostatType::STOCHASTIC_RESCALING;
}

/**
 * @brief get the isotropy of the manostat
 *
 * @return Isotropy
 */
Isotropy StochasticRescalingManostat::getIsotropy() const
{
    return Isotropy::ISOTROPIC;
}

/**
 * @brief get the isotropy of the manostat
 *
 * @return Isotropy
 */
Isotropy SemiIsotropicStochasticRescalingManostat::getIsotropy() const
{
    return Isotropy::SEMI_ISOTROPIC;
}

/**
 * @brief get the isotropy of the manostat
 *
 * @return Isotropy
 */
Isotropy AnisotropicStochasticRescalingManostat::getIsotropy() const
{
    return Isotropy::ANISOTROPIC;
}

/**
 * @brief get the isotropy of the manostat
 *
 * @return Isotropy
 */
Isotropy FullAnisotropicStochasticRescalingManostat::getIsotropy() const
{
    return Isotropy::FULL_ANISOTROPIC;
}
