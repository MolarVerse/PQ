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

#include <algorithm>   // for __for_each_fn
#include <cmath>       // for exp, pow, sqrt

#include "constants/conversionFactors.hpp"   // for _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_
#include "constants/internalConversionFactors.hpp"   // for _PRESSURE_FACTOR_
#include "globalTimer.hpp"
#include "manostatSettings.hpp"     // for ManostatType, Isotropy
#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "timingsSettings.hpp"      // for TimingsSettings

using namespace linearAlgebra;
using namespace manostat;
using namespace settings;
using namespace molsys;
using namespace physicalData;
using namespace exc;
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
      _dt(other._dt),
      _fixedAxis(other._fixedAxis)
{
}

/**
 * @brief copy assignment operator for Stochastic Rescaling Manostat
 *
 * @param other
 * @return StochasticRescalingManostat&
 */
StochasticRescalingManostat &StochasticRescalingManostat::operator=(
    const StochasticRescalingManostat &other
)
{
    if (this != &other)
    {
        Manostat::operator=(other);
        _tau             = other._tau;
        _compressibility = other._compressibility;
        _dt              = other._dt;
        _fixedAxis       = other._fixedAxis;
    }
    return *this;
}

/**
 * @brief Construct a new Stochastic Rescaling Manostat:: Stochastic Rescaling
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 * @param anisotropicAxis
 * @param isotropicAxes
 * @param fixedAxis
 * @return SemiIsotropicStochasticRescalingManostat::
 */
SemiIsotropicStochasticRescalingManostat::
    SemiIsotropicStochasticRescalingManostat(
        const double               targetPressure,
        const double               tau,
        const double               compressibility,
        const size_t               anisotropicAxis,
        const std::vector<size_t> &isotropicAxes,
        const FixedAxis            fixedAxis
    )
    : StochasticRescalingManostat(
          targetPressure,
          tau,
          compressibility,
          fixedAxis
      ),
      _2DAnisotropicAxis(anisotropicAxis),
      _2DIsotropicAxes(isotropicAxes)
{
}

/**
 * @brief Construct a new Stochastic Rescaling Manostat:: Stochastic Rescaling
 * Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 * @param fixedAxis
 */
StochasticRescalingManostat::StochasticRescalingManostat(
    const double    targetPressure,
    const double    tau,
    const double    compressibility,
    const FixedAxis fixedAxis
)
    : Manostat(targetPressure),
      _tau(tau),
      _compressibility(compressibility),
      _dt(TimingsSettings::getTimeStep()),
      _fixedAxis(fixedAxis)
{
}

/**
 * @brief apply Stochastic Rescaling manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void StochasticRescalingManostat::applyManostat(
    molsys::SimulationBox      &simBox,
    physicalData::PhysicalData &physicalData
)
{
    auto _ = scopedTimer(TimerId::Thermostat, "Stochastic Rescaling");

    calculatePressure(simBox, physicalData);

    const auto mu = calculateMu(simBox.getVolume());

    // Reconstruction temporarily unwraps atoms. Molecule::scale() below wraps
    // every position into the resized box.
    auto reconstructMolecule = [&simBox](auto &molecule)
    { molecule.reconstructAtomsAroundCenterOfMass(simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), reconstructMolecule);

    simBox.scaleBox(mu);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulRadiusCutOff(ExceptionType::ManostatError);

    auto scalePositions = [&mu, &simBox](auto &molecule)
    { molecule.scale(mu, simBox.getBox()); };

    auto scaleVelocities = [&mu, &simBox](auto &molecule)
    { molecule.scaleVelocity(inverse(mu), simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), scalePositions);
    std::ranges::for_each(simBox.getMolecules(), scaleVelocities);
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat
 * (isotropic)
 *
 * @details If a fixed axis is specified, that axis is not scaled (mu = 1.0)
 * and the remaining axes are scaled isotropically with stochastic coupling
 *
 * @param volume
 * @return Vec3D
 */
tensor3D StochasticRescalingManostat::calculateMu(const double volume)
{
    using enum FixedAxis;

    const auto compress = _compressibility * _dt / _tau;
    const auto kb       = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

    // 2D pressure coupling
    if (_fixedAxis != NONE)
    {
        const auto fixedAxisIndex = static_cast<size_t>(_fixedAxis) - 1;
        const auto p_xyz          = diagonal(_pressureTensor);

        // Calculate average pressure of non-fixed axes
        double p_avg = 0.0;
        for (size_t i = 0; i < 3; ++i)
            if (i != fixedAxisIndex)
                p_avg += p_xyz[i];
        p_avg /= 2.0;

        auto stochasticFactor  = 2.0 * kT * compress / volume;
        stochasticFactor      *= PRESSURE_FACTOR;
        stochasticFactor       = ::sqrt(stochasticFactor) * random;

        const auto deltaP = _targetPressure - p_avg;

        // 2D isotropic scaling
        constexpr auto dimension = 2.0;
        const auto     mu_2D =
            ::exp((-compress * deltaP + stochasticFactor) / dimension);

        Vec3D mu = {1.0, 1.0, 1.0};
        for (size_t i = 0; i < 3; ++i)
            if (i != fixedAxisIndex)
                mu[i] = mu_2D;

        return diagonalMatrix(mu);
    }

    // 3D pressure coupling
    auto stochasticFactor  = 2.0 * kT * compress / volume;
    stochasticFactor      *= PRESSURE_FACTOR;
    stochasticFactor       = ::sqrt(stochasticFactor) * random;

    const auto deltaP = _targetPressure - _pressure;

    // TODO: check how to generalize this!
    constexpr auto dimension = 3.0;

    return diagonalMatrix(
        ::exp((-compress * deltaP + stochasticFactor) / dimension)
    );
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
    const auto kb       = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

    auto stochasticFactor = 1.0 / _pressureTensor.size * kT * compress / volume;
    stochasticFactor *= PRESSURE_FACTOR;

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
 * @details If a fixed axis is specified, that axis is not scaled (mu = 1.0)
 * and the other axes are scaled independently with stochastic coupling
 *
 * @param volume
 * @return Vec3D
 */
tensor3D AnisotropicStochasticRescalingManostat::calculateMu(
    const double volume
)
{
    using enum FixedAxis;

    const auto compress = _compressibility * _dt / _tau;
    const auto kb       = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

    auto stochasticFactor = 2.0 / _pressureTensor.size * kT * compress / volume;
    stochasticFactor *= PRESSURE_FACTOR;
    stochasticFactor  = ::sqrt(stochasticFactor) * random;

    const auto deltaP = _targetPressure - diagonal(_pressureTensor);

    auto mu =
        exp(-compress * (deltaP) / _pressureTensor.size + stochasticFactor);

    // 2D anisotropic: fix one axis
    if (_fixedAxis != NONE)
    {
        const auto fixedAxisIndex = static_cast<size_t>(_fixedAxis) - 1;
        mu[fixedAxisIndex]        = 1.0;
    }

    return diagonalMatrix(mu);
}

/**
 * @brief calculate mu as scaling factor for Stochastic Rescaling manostat (full
 * anisotropic including angles)
 *
 * @details If a fixed axis is specified, the corresponding row and column
 * are zeroed (no coupling with other axes) and the diagonal is set to 1.0
 *
 * @param volume
 * @return tensor3D
 */
tensor3D FullAnisotropicStochasticRescalingManostat::calculateMu(
    const double volume
)
{
    using enum FixedAxis;

    const auto compress = _compressibility * _dt / _tau;
    const auto kb       = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

    const auto kT     = kb * ThermostatSettings::getActualTargetTemperature();
    const auto random = _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

    auto stochasticFactor = 2.0 / _pressureTensor.size * kT * compress / volume;
    stochasticFactor *= PRESSURE_FACTOR;
    stochasticFactor  = ::sqrt(stochasticFactor) * random;

    const auto deltaP = diagonalMatrix(_targetPressure) - _pressureTensor;
    auto       mu =
        expPade(-compress * deltaP / _pressureTensor.size + stochasticFactor);

    // 2D full anisotropic: fix one axis and remove its coupling
    if (_fixedAxis != NONE)
    {
        const auto fixedAxisIndex = static_cast<size_t>(_fixedAxis) - 1;

        // Zero out the row and column of the fixed axis
        for (size_t i = 0; i < 3; ++i)
        {
            mu[fixedAxisIndex][i] = 0.0;
            mu[i][fixedAxisIndex] = 0.0;
        }
        // Set diagonal to 1.0 (no scaling)
        mu[fixedAxisIndex][fixedAxisIndex] = 1.0;
    }

    rotateMu(mu);

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
