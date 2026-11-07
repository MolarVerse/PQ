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

#include "berendsenManostat.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

#include "globalTimer.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace manostat
{
    /**
     * @brief calculate the pre-factor for Berendsen manostat scaling
     *
     * @return Compressibility
     */
    Compressibility BerendsenManostat::_calcPreFactor() const
    {
        const auto compressibility = _getSettings().getCompressibility();
        const auto tau             = _getSettings().getTau();
        const auto timeStep        = settings::TimingsSettings::getTimeStep();

        return compressibility * timeStep / tau;
    }

    /**
     * @brief apply Berendsen manostat for NPT ensemble
     *
     * @param simulationBox
     * @param physicalData
     */
    void BerendsenManostat::applyManostat(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData
    )
    {
        auto _ = scopedTimer(TimerId::Manostat, "Berendsen");

        calculatePressure(simulationBox, physicalData);

        const auto mu = calculateMu();

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

        const auto &box        = simulationBox.getBox();
        const auto  muSimSpace = _toSimSpaceScaling(mu, box);

        simulationBox.scaleBox(mu);
        simulationBox.checkCoulRadiusCutOff(ExceptionType::ManostatError);

        physicalData.setVolume(simulationBox.getVolume());
        physicalData.setDensity(simulationBox.getDensity());

        auto scaleMolecule = [&muSimSpace, &box](auto &molecule)
        { molecule.scale(muSimSpace, box); };

        std::ranges::for_each(simulationBox.getMolecules(), scaleMolecule);
    }

    /**
     * @brief calculate mu as scaling factor for Berendsen manostat (isotropic)
     *
     * @details If fixed axes are specified, those axes are not scaled (mu
     * = 1.0) and the remaining axes are scaled isotropically
     *
     * @return linalg::tensor3D
     */
    linalg::tensor3D BerendsenManostat::calculateMu() const
    {
        const auto targetPressure = _getSettings().getTargetPressure() * bar;
        const auto fixedAxis      = _getSettings().getFixedAxis();
        if (fixedAxis == FixedAxis::ALL)
            return linalg::diagonalMatrix(linalg::Vec3D{1.0, 1.0, 1.0});

        const auto preFactor = _calcPreFactor();
        const auto p_xyz     = diagonal(_getPressureTensor());

        size_t   numFree = 0;
        Pressure p_avg{0.0 * Pa};

        for (size_t axis = 0; axis < 3; ++axis)
        {
            if (!isAxisFixed(fixedAxis, axis))
            {
                p_avg += p_xyz[axis];
                ++numFree;
            }
        }

        p_avg /= static_cast<double>(numFree);

        const auto deltaP = targetPressure - p_avg;

        double mu_scaled = ::pow(
            1.0 - (preFactor * deltaP).numerical_value_in(one),
            1.0 / static_cast<double>(numFree)
        );

        linalg::Vec3D mu = {1.0, 1.0, 1.0};
        for (size_t i = 0; i < 3; ++i)
        {
            if (!isAxisFixed(fixedAxis, i))
                mu[i] = mu_scaled;
        }

        return linalg::diagonalMatrix(mu);
    }

    /**
     * @brief calculate mu as scaling factor for Berendsen manostat
     * (semi-isotropic)
     *
     * @details _2DIsotropicAxes[0] and _2DIsotropicAxes[1] are the indices of
     * the isotropic coupled axes and _2DAnisotropicAxis is the index of the
     * anisotropic axis
     *
     * @return linalg::tensor3D
     */
    linalg::tensor3D SemiIsotropicBerendsenManostat::calculateMu() const
    {
        const auto p_xyz           = diagonal(_getPressureTensor());
        const auto isotropy        = _getSettings().getIsotropy();
        const auto anisotropicAxis = get2DAnisotropicAxis(isotropy);
        const auto isotropicAxes   = get2DIsotropicAxes(isotropy);
        const auto p_x             = p_xyz[isotropicAxes[0]];
        const auto p_y             = p_xyz[isotropicAxes[1]];
        const auto p_xy            = (p_x + p_y) / 2.0;
        const auto p_z             = p_xyz[anisotropicAxis];

        const auto dimension =
            3.0 -
            std::popcount(static_cast<unsigned>(_getSettings().getFixedAxis()));
        const auto preFactor = _calcPreFactor() / dimension;

        const auto targetPressure = _getSettings().getTargetPressure() * bar;

        const double mu_xy = ::sqrt(
            1.0 -
            (2.0 * preFactor * (targetPressure - p_xy)).numerical_value_in(one)
        );
        const double mu_z =
            isAxisFixed(_getSettings().getFixedAxis(), anisotropicAxis)
                ? 1.0
                : (1.0 -
                   (preFactor * (targetPressure - p_z)).numerical_value_in(one)
                  );

        linalg::Vec3D mu;

        mu[isotropicAxes[0]] = mu_xy;
        mu[isotropicAxes[1]] = mu_xy;
        mu[anisotropicAxis]  = mu_z;

        return linalg::diagonalMatrix(mu);
    }

    /**
     * @brief calculate mu as scaling factor for Berendsen manostat
     * (anisotropic)
     *
     * @details If fixed axes are specified, those axes are not scaled (mu
     * = 1.0) and the other axes are scaled independently
     *
     * @return linalg::tensor3D
     */
    linalg::tensor3D AnisotropicBerendsenManostat::calculateMu() const
    {
        const auto pxyz           = diagonal(_getPressureTensor());
        const auto targetPressure = _getSettings().getTargetPressure() * bar;
        const auto fixedAxis      = _getSettings().getFixedAxis();
        const auto dimension =
            3.0 - std::popcount(static_cast<unsigned>(fixedAxis));
        const auto preFactor = _calcPreFactor() / dimension;

        auto mu = 1.0 - ((Pressure3D{targetPressure} - pxyz) * preFactor);

        for (size_t i = 0; i < 3; ++i)
        {
            if (isAxisFixed(fixedAxis, i))
                mu[i] = 1.0;
        }

        return linalg::diagonalMatrix(mu.numerical_value_in(one));
    }

    /**
     * @brief calculate mu as scaling factor for Berendsen manostat (full
     * anisotropic including angles)
     *
     * @details If fixed axes are specified, the corresponding rows and columns
     * are zeroed (no coupling with other axes) and the diagonals are set to 1.0
     *
     * @return linalg::tensor3D
     */
    linalg::tensor3D FullAnisotropicBerendsenManostat::calculateMu() const
    {
        const auto pTarget =
            linalg::diagonalMatrix(_getSettings().getTargetPressure()) * bar;
        const auto &fixedAxis = _getSettings().getFixedAxis();
        const auto  dimension =
            3.0 - std::popcount(static_cast<unsigned>(fixedAxis));
        const auto preFactor = _calcPreFactor() / dimension;
        const auto kronecker = linalg::kroneckerDeltaMatrix<double>();

        auto mu = kronecker - ((pTarget - _getPressureTensor()) * preFactor)
                                  .numerical_value_in(one);

        for (size_t k = 0; k < 3; ++k)
        {
            if (isAxisFixed(fixedAxis, k))
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

}   // namespace manostat
