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
#include "enums/manostat.hpp"
#include "globalTimer.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "timingsSettings.hpp"

namespace manostat
{

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

        _validateScaling(simulationBox, mu);

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
        simulationBox.checkCoulRadiusCutOff(ExceptionType::ManostatError);

        physicalData.setVolume(simulationBox.getVolume());
        physicalData.setDensity(simulationBox.getDensity());

        auto scalePositions = [&mu, &simulationBox](auto &molecule)
        { molecule.scale(mu, simulationBox.getBox()); };

        auto scaleVelocities = [&mu, &simulationBox](auto &molecule)
        { molecule.scaleVelocity(inverse(mu), simulationBox.getBox()); };

        std::ranges::for_each(simulationBox.getMolecules(), scalePositions);
        std::ranges::for_each(simulationBox.getMolecules(), scaleVelocities);

        physicalData.calculateKinetics(simulationBox);
        physicalData.calculateTemperature(simulationBox);
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
        const auto fixedAxis       = _getSettings().getFixedAxis();
        const auto compressibility = _getSettings().getCompressibility();
        const auto tau             = _getSettings().getTauManostat();

        if (fixedAxis == FixedAxis::ALL)
            return diagonalMatrix(linalg::Vec3D{1.0, 1.0, 1.0});

        const auto compress =
            compressibility * settings::TimingsSettings::getTimeStep() / tau;
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();

        const auto random =
            _randomNumberGenerator.getNormalDistribution(0.0, 1.0);

        auto stochasticFactor  = 2.0 * thermalEnergy * compress / volume;
        stochasticFactor      *= PRESSURE_FACTOR;
        stochasticFactor       = ::sqrt(stochasticFactor) * random;

        if (fixedAxis == FixedAxis::NONE)
        {
            const auto deltaP =
                _getSettings().getTargetPressure() - _getPressure();
            constexpr auto dimension = 3.0;

            return linalg::diagonalMatrix(
                ::exp(((-compress * deltaP) + stochasticFactor) / dimension)
            );
        }

        const auto p_xyz = diagonal(_getPressureTensor());

        size_t numFree = 0;
        double p_avg   = 0.0;

        for (size_t axis = 0; axis < 3; ++axis)
        {
            if (!_hasFixedAxis(axis))   // use the fixedAxis from settings
            {
                p_avg += p_xyz[axis];
                ++numFree;
            }
        }

        p_avg /= static_cast<double>(numFree);

        const auto deltaP    = _getSettings().getTargetPressure() - p_avg;
        const auto dimension = static_cast<double>(numFree);

        const auto mu_scaled =
            ::exp(((-compress * deltaP) + stochasticFactor) / dimension);

        linalg::Vec3D mu = {1.0, 1.0, 1.0};
        for (size_t i = 0; i < 3; ++i)
        {
            if (!_hasFixedAxis(i))
                mu[i] = mu_scaled;
        }

        return diagonalMatrix(mu);
    }

    /**
     * @brief Check if a specific axis is fixed
     *
     * @param index Axis index (0 for x, 1 for y, 2 for z)
     * @return true if the axis is fixed, false otherwise
     */
    bool StochasticRescalingManostat::_hasFixedAxis(size_t index) const
    {
        return isAxisFixed(_getSettings().getFixedAxis(), index);
    }

    /**
     * @brief Get the random number generator
     *
     * @return rng::RandomNumberGenerator&
     */
    rng::RandomNumberGenerator &StochasticRescalingManostat::_getRNG()
    {
        return _randomNumberGenerator;
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
        const auto compress = _getSettings().getCompressibility() *
                              settings::TimingsSettings::getTimeStep() /
                              _getSettings().getTauManostat();
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();

        auto stochasticFactor =
            1.0 / linalg::tensor3D::size * thermalEnergy * compress / volume;
        stochasticFactor *= PRESSURE_FACTOR;

        // Independent area and height noise (Bernetti and Bussi, Eq. 9).
        const auto stochasticFactor_xy =
            ::sqrt(4.0 * stochasticFactor) *
            _getRNG().getNormalDistribution(0.0, 1.0);
        const auto stochasticFactor_z =
            ::sqrt(2.0 * stochasticFactor) *
            _getRNG().getNormalDistribution(0.0, 1.0);

        const auto isotropy        = _getSettings().getIsotropy();
        const auto p_xyz           = diagonal(_getPressureTensor());
        const auto isotropicAxes   = get2DIsotropicAxes(isotropy);
        const auto anisotropicAxis = get2DAnisotropicAxis(isotropy);
        const auto p_x             = p_xyz[isotropicAxes[0]];
        const auto p_y             = p_xyz[isotropicAxes[1]];
        const auto p_xy            = (p_x + p_y) / 2.0;
        const auto p_z             = p_xyz[anisotropicAxis];

        const auto targetPressure = _getSettings().getTargetPressure();
        const auto deltaPxy       = targetPressure - p_xy;
        const auto deltaPz        = targetPressure - p_z;

        const auto mu_xy =
            ::exp((-compress * deltaPxy / 3.0) + (stochasticFactor_xy / 2.0));
        const auto mu_z =
            _hasFixedAxis(anisotropicAxis)
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
        const auto compress = _getSettings().getCompressibility() *
                              settings::TimingsSettings::getTimeStep() /
                              _getSettings().getTauManostat();
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();

        auto stochasticFactor =
            2.0 / linalg::tensor3D::size * thermalEnergy * compress / volume;
        stochasticFactor *= PRESSURE_FACTOR;
        stochasticFactor  = ::sqrt(stochasticFactor);

        const auto deltaP =
            _getSettings().getTargetPressure() - diagonal(_getPressureTensor());

        auto mu = linalg::Vec3D{1.0, 1.0, 1.0};
        for (size_t i = 0; i < 3; ++i)
        {
            if (!_hasFixedAxis(i))
            {
                mu[i] = ::exp(
                    (-compress * deltaP[i] / linalg::tensor3D::size) +
                    (stochasticFactor *
                     _getRNG().getNormalDistribution(0.0, 1.0))
                );
            }
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
        const auto compress = _getSettings().getCompressibility() *
                              settings::TimingsSettings::getTimeStep() /
                              _getSettings().getTauManostat();
        const auto boltzmannConstant = BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL;

        const auto thermalEnergy =
            boltzmannConstant *
            settings::ThermostatSettings::getActualTargetTemperature();

        auto stochasticFactor =
            2.0 / linalg::tensor3D::size * thermalEnergy * compress / volume;
        stochasticFactor *= PRESSURE_FACTOR;
        stochasticFactor  = ::sqrt(stochasticFactor);

        const auto deltaP =
            linalg::diagonalMatrix(_getSettings().getTargetPressure()) -
            _getPressureTensor();

        auto strain = -compress * deltaP / linalg::tensor3D::size;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                if (_hasFixedAxis(i) || _hasFixedAxis(j))
                    strain[i][j] = 0.0;
                else
                    strain[i][j] += stochasticFactor *
                                    _getRNG().getNormalDistribution(0.0, 1.0);
            }
        }

        auto mu = expPade(strain);

        rotateMu(mu);

        return mu;
    }

}   // namespace manostat
