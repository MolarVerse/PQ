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
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"

namespace manostat
{

    /**
     * @brief Construct a new Berendsen Manostat:: Berendsen Manostat object
     *
     * @param targetPressure
     * @param tau
     * @param compressibility
     * @param fixedAxis
     */
    BerendsenManostat::BerendsenManostat(
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
     * @brief Construct a new Berendsen Manostat:: Berendsen Manostat object
     *
     * @param targetPressure
     * @param tau
     * @param compressibility
     * @param isotropy
     * @param fixedAxis
     */
    SemiIsotropicBerendsenManostat::SemiIsotropicBerendsenManostat(
        double    targetPressure,
        double    tau,
        double    compressibility,
        Isotropy  isotropy,
        FixedAxis fixedAxis
    )
        : BerendsenManostat(targetPressure, tau, compressibility, fixedAxis),
          _isotropy(isotropy)
    {
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

        simulationBox.scaleBox(mu);

        physicalData.setVolume(simulationBox.getVolume());
        physicalData.setDensity(simulationBox.getDensity());

        auto scaleMolecule = [&mu, &simulationBox](auto &molecule)
        { molecule.scale(mu, simulationBox.getBox()); };

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
        if (_fixedAxis == FixedAxis::ALL)
            return linalg::diagonalMatrix(linalg::Vec3D{1.0, 1.0, 1.0});

        const auto preFactor = _compressibility * _dt / _tau;
        const auto p_xyz     = diagonal(_pressureTensor);

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

        const auto deltaP = _targetPressure - p_avg;

        double mu_scaled = 1.0;
        if (numFree == 3)
            mu_scaled = ::cbrt(1.0 - (preFactor * deltaP));
        else if (numFree == 2)
            mu_scaled = ::sqrt(1.0 - (preFactor * deltaP));
        else if (numFree == 1)
            mu_scaled = 1.0 - (preFactor * deltaP);

        linalg::Vec3D mu = {1.0, 1.0, 1.0};
        for (size_t i = 0; i < 3; ++i)
        {
            if (!isAxisFixed(_fixedAxis, i))
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
        if (_fixedAxis == FixedAxis::ALL)
            return linalg::diagonalMatrix(1.0);

        const auto p_xyz           = diagonal(_pressureTensor);
        const auto anisotropicAxis = get2DAnisotropicAxis(_isotropy);
        const auto isotropicAxes   = get2DIsotropicAxes(_isotropy);
        const auto p_x             = p_xyz[isotropicAxes[0]];
        const auto p_y             = p_xyz[isotropicAxes[1]];
        const auto p_xy            = (p_x + p_y) / 2.0;
        const auto p_z             = p_xyz[anisotropicAxis];

        const auto dimension =
            3.0 - std::popcount(static_cast<unsigned>(_fixedAxis));
        const auto preFactor = _compressibility * _dt / (_tau * dimension);

        const double mu_xy =
            ::sqrt(1.0 - (2.0 * preFactor * (_targetPressure - p_xy)));
        const double mu_z = isAxisFixed(_fixedAxis, anisotropicAxis)
                                ? 1.0
                                : (1.0 - (preFactor * (_targetPressure - p_z)));

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
        if (_fixedAxis == FixedAxis::ALL)
            return linalg::diagonalMatrix(1.0);

        const auto pxyz = diagonal(_pressureTensor);
        const auto dimension =
            3.0 - std::popcount(static_cast<unsigned>(_fixedAxis));
        const auto preFactor = _compressibility * _dt / (_tau * dimension);

        auto mu = 1.0 - preFactor * (_targetPressure - pxyz);

        for (size_t i = 0; i < 3; ++i)
        {
            if (isAxisFixed(_fixedAxis, i))
                mu[i] = 1.0;
        }

        return linalg::diagonalMatrix(mu);
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
        if (_fixedAxis == FixedAxis::ALL)
            return linalg::diagonalMatrix(1.0);

        const auto pTarget = linalg::diagonalMatrix(_targetPressure);
        const auto dimension =
            3.0 - std::popcount(static_cast<unsigned>(_fixedAxis));
        const auto preFactor = _compressibility * _dt / (_tau * dimension);
        const auto kronecker = linalg::kroneckerDeltaMatrix<double>();

        auto mu = kronecker - preFactor * (pTarget - _pressureTensor);

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
    double BerendsenManostat::getTau() const { return _tau; }

    /**
     * @brief get compressibility
     *
     * @return double
     */
    double BerendsenManostat::getCompressibility() const
    {
        return _compressibility;
    }

    /**
     * @brief get the manostat type
     *
     * @return ManostatType
     */
    ManostatType BerendsenManostat::getManostatType() const
    {
        return ManostatType::BERENDSEN;
    }

}   // namespace manostat
