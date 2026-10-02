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

#include <algorithm>
#include <cmath>

#include "constants/conversionFactors.hpp"
#include "constants/internalConversionFactors.hpp"
#include "globalTimer.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "timingsSettings.hpp"

namespace manostat
{

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
     * @brief Construct a new Stochastic Rescaling Manostat:: Stochastic
     * Rescaling
     *
     * @param targetPressure
     * @param tau
     * @param compressibility
     * @param isotropy
     * @param fixedAxis
     */
    SemiIsotropicStochasticRescalingManostat::
        SemiIsotropicStochasticRescalingManostat(
            double    targetPressure,
            double    tau,
            double    compressibility,
            Isotropy  isotropy,
            FixedAxis fixedAxis
        )
        : StochasticRescalingManostat(
              targetPressure,
              tau,
              compressibility,
              fixedAxis
          ),
          _isotropy(isotropy)
    {
    }

    /**
     * @brief Construct a new Stochastic Rescaling Manostat:: Stochastic
     * Rescaling Manostat object
     *
     * @param targetPressure
     * @param tau
     * @param compressibility
     * @param fixedAxis
     */
    StochasticRescalingManostat::StochasticRescalingManostat(
        double    targetPressure,
        double    tau,
        double    compressibility,
        FixedAxis fixedAxis
    )
        : Manostat(targetPressure),
          _tau(tau),
          _compressibility(compressibility),
          _dt(settings::TimingsSettings::getTimeStep()),
          _fixedAxis(fixedAxis)
    {
    }

    /**
     * @brief apply Stochastic Rescaling manostat for NPT ensemble
     *
     * @param simulationBox
     * @param physicalData
     */
    void StochasticRescalingManostat::applyManostat(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData
    )
    {
        auto _ = scopedTimer(TimerId::Manostat, "Stochastic Rescaling");

        calculatePressure(simulationBox, physicalData);

        const auto mu = calculateMu(simulationBox.getVolume());

        // Reconstruction temporarily unwraps atoms. Molecule::scale() below
        // wraps every position into the resized box.
        auto reconstructMolecule = [&simulationBox](auto &molecule)
        {
            molecule.reconstructAtomsAroundCenterOfMass(simulationBox.getBox());
        };

        std::ranges::for_each(
            simulationBox.getMolecules(),
            reconstructMolecule
        );

        simulationBox.scaleBox(mu);

        physicalData.setVolume(simulationBox.getVolume());
        physicalData.setDensity(simulationBox.getDensity());

        simulationBox.checkCoulRadiusCutOff(ExceptionType::ManostatError);

        auto scalePositions = [&mu, &simulationBox](auto &molecule)
        { molecule.scale(mu, simulationBox.getBox()); };

        auto scaleVelocities = [&mu, &simulationBox](auto &molecule)
        { molecule.scaleVelocity(inverse(mu), simulationBox.getBox()); };

        std::ranges::for_each(simulationBox.getMolecules(), scalePositions);
        std::ranges::for_each(simulationBox.getMolecules(), scaleVelocities);
    }

    /**
     * @brief calculate mu as scaling factor for Stochastic Rescaling manostat
     * (isotropic)
     *
     * @details If a fixed axis is specified, that axis is not scaled (mu = 1.0)
     * and the remaining axes are scaled isotropically with stochastic coupling
     *
     * @param volume
     * @return linalg::Vec3D
     */
    linalg::tensor3D StochasticRescalingManostat::calculateMu(double volume)
    {
        if (_fixedAxis == FixedAxis::ALL)
            return diagonalMatrix(linalg::Vec3D{1.0, 1.0, 1.0});

        const auto compress          = _compressibility * _dt / _tau;
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();

        const auto random =
            _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

        auto stochasticFactor  = 2.0 * thermalEnergy * compress / volume;
        stochasticFactor      *= PRESSURE_FACTOR;
        stochasticFactor       = ::sqrt(stochasticFactor) * random;

        if (_fixedAxis == FixedAxis::NONE)
        {
            const auto     deltaP    = _targetPressure - _pressure;
            constexpr auto dimension = 3.0;

            return linalg::diagonalMatrix(
                ::exp(((-compress * deltaP) + stochasticFactor) / dimension)
            );
        }

        const auto p_xyz = diagonal(_pressureTensor);

        size_t numFree = 0;
        double p_avg   = 0.0;

        for (size_t axis = 0; axis < 3; ++axis)
        {
            if (!isAxisFixed(_fixedAxis, axis))
            {
                p_avg += p_xyz[axis];
                ++numFree;
            }
        }

        p_avg /= static_cast<double>(numFree);

        const auto deltaP    = _targetPressure - p_avg;
        const auto dimension = static_cast<double>(numFree);

        const auto mu_scaled =
            ::exp(((-compress * deltaP) + stochasticFactor) / dimension);

        linalg::Vec3D mu = {1.0, 1.0, 1.0};
        for (size_t i = 0; i < 3; ++i)
        {
            if (!isAxisFixed(_fixedAxis, i))
                mu[i] = mu_scaled;
        }

        return diagonalMatrix(mu);
    }

    /**
     * @brief calculate mu as scaling factor for Stochastic Rescaling manostat
     * (semi-isotropic)
     *
     * @param volume
     * @return linalg::Vec3D
     */
    linalg::tensor3D SemiIsotropicStochasticRescalingManostat::calculateMu(
        double volume
    )
    {
        const auto compress          = _compressibility * _dt / _tau;
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();
        const auto random =
            _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

        auto stochasticFactor =
            1.0 / linalg::tensor3D::size * thermalEnergy * compress / volume;
        stochasticFactor *= PRESSURE_FACTOR;

        const auto stochasticFactor_xy =
            ::sqrt(4.0 * stochasticFactor) * random;
        const auto stochasticFactor_z = ::sqrt(2.0 * stochasticFactor) * random;

        const auto p_xyz           = diagonal(_pressureTensor);
        const auto isotropicAxes   = get2DIsotropicAxes(_isotropy);
        const auto anisotropicAxis = get2DAnisotropicAxis(_isotropy);
        const auto p_x             = p_xyz[isotropicAxes[0]];
        const auto p_y             = p_xyz[isotropicAxes[1]];
        const auto p_xy            = (p_x + p_y) / 2.0;
        const auto p_z             = p_xyz[anisotropicAxis];

        const auto deltaPxy = _targetPressure - p_xy;
        const auto deltaPz  = _targetPressure - p_z;

        const auto mu_xy =
            ::exp((-compress * deltaPxy / 3.0) + (stochasticFactor_xy / 2.0));
        const auto mu_z =
            isAxisFixed(_fixedAxis, anisotropicAxis)
                ? 1.0
                : ::exp((-compress * deltaPz / 3.0) + stochasticFactor_z);

        linalg::Vec3D mu;

        mu[isotropicAxes[0]] = mu_xy;
        mu[isotropicAxes[1]] = mu_xy;
        mu[anisotropicAxis]  = mu_z;

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
     * @return linalg::Vec3D
     */
    linalg::tensor3D AnisotropicStochasticRescalingManostat::calculateMu(
        double volume
    )
    {
        const auto compress          = _compressibility * _dt / _tau;
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();
        const auto random =
            _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

        auto stochasticFactor =
            2.0 / linalg::tensor3D::size * thermalEnergy * compress / volume;
        stochasticFactor *= PRESSURE_FACTOR;
        stochasticFactor  = ::sqrt(stochasticFactor) * random;

        const auto deltaP = _targetPressure - diagonal(_pressureTensor);

        auto mu =
            exp(-compress * (deltaP) / linalg::tensor3D::size + stochasticFactor
            );

        for (size_t i = 0; i < 3; ++i)
        {
            if (isAxisFixed(_fixedAxis, i))
                mu[i] = 1.0;
        }

        return diagonalMatrix(mu);
    }

    /**
     * @brief calculate mu as scaling factor for Stochastic Rescaling manostat
     * (full anisotropic including angles)
     *
     * @details If fixed axes are specified, the corresponding rows and columns
     * are zeroed (no coupling with other axes) and the diagonals are set to 1.0
     *
     * @param volume
     * @return linalg::tensor3D
     */
    linalg::tensor3D FullAnisotropicStochasticRescalingManostat::calculateMu(
        double volume
    )
    {
        const auto compress          = _compressibility * _dt / _tau;
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();
        const auto random =
            _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

        auto stochasticFactor =
            2.0 / linalg::tensor3D::size * thermalEnergy * compress / volume;
        stochasticFactor *= PRESSURE_FACTOR;
        stochasticFactor  = ::sqrt(stochasticFactor) * random;

        const auto deltaP =
            linalg::diagonalMatrix(_targetPressure) - _pressureTensor;

        auto mu = expPade(
            -compress * deltaP / linalg::tensor3D::size + stochasticFactor
        );

        for (size_t k = 0; k < 3; ++k)
        {
            if (isAxisFixed(_fixedAxis, k))
            {
                for (size_t i = 0; i < 3; ++i)
                {
                    mu[k][i] = 0.0;
                    mu[i][k] = 0.0;
                }
                mu[k][k] = 1.0;
            }
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

}   // namespace manostat
