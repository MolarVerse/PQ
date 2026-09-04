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

#include <algorithm>   // for __for_each_fn, for_each
#include <cmath>       // for cbrt

#include "globalTimer.hpp"
#include "manostatSettings.hpp"   // for ManostatType, Isotropy
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "timingsSettings.hpp"    // for TimingsSettings

using namespace linearAlgebra;
using namespace settings;
using namespace manostat;
using namespace exc;
using namespace molsys;
using namespace physicalData;

/**
 * @brief Construct a new Berendsen Manostat:: Berendsen Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 * @param fixedAxis
 */
BerendsenManostat::BerendsenManostat(
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
 * @brief Construct a new Berendsen Manostat:: Berendsen Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 * @param anisotropicAxis
 * @param isotropicAxes
 * @param fixedAxis
 */
SemiIsotropicBerendsenManostat::SemiIsotropicBerendsenManostat(
    const double               targetPressure,
    const double               tau,
    const double               compressibility,
    const size_t               anisotropicAxis,
    const std::vector<size_t> &isotropicAxes,
    const FixedAxis            fixedAxis
)
    : BerendsenManostat(targetPressure, tau, compressibility, fixedAxis),
      _2DAnisotropicAxis(anisotropicAxis),
      _2DIsotropicAxes(isotropicAxes)
{
}

/**
 * @brief apply Berendsen manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void BerendsenManostat::applyManostat(
    SimulationBox &simBox,
    PhysicalData  &physicalData
)
{
    auto _ = scopedTimer(TimerId::Manostat, "Berendsen");

    calculatePressure(simBox, physicalData);

    const auto mu = calculateMu();

    // Reconstruction temporarily unwraps atoms. Molecule::scale() below wraps
    // every position into the resized box.
    auto reconstructMolecule = [&simBox](auto &molecule)
    { molecule.reconstructAtomsAroundCenterOfMass(simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), reconstructMolecule);

    simBox.scaleBox(mu);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulRadiusCutOff(ExceptionType::ManostatError);

    auto scaleMolecule = [&mu, &simBox](auto &molecule)
    { molecule.scale(mu, simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), scaleMolecule);
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (isotropic)
 *
 * @details If a fixed axis is specified, that axis is not scaled (mu = 1.0)
 * and the remaining axes are scaled isotropically
 *
 * @return tensor3D
 */
tensor3D BerendsenManostat::calculateMu() const
{
    using enum FixedAxis;

    const auto preFactor = _compressibility * _dt / _tau;

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

        // Scale factor for non-fixed axes
        const auto mu_2D = ::sqrt(1.0 - preFactor * (_targetPressure - p_avg));

        Vec3D mu = {1.0, 1.0, 1.0};
        for (size_t i = 0; i < 3; ++i)
            if (i != fixedAxisIndex)
                mu[i] = mu_2D;

        return diagonalMatrix(mu);
    }

    // 3D pressure coupling
    const auto p         = trace(_pressureTensor) / 3.0;
    const auto mu_scalar = ::cbrt(1.0 - preFactor * (_targetPressure - p));

    return diagonalMatrix(mu_scalar);
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (semi-isotropic)
 *
 * @details _2DIsotropicAxes[0] and _2DIsotropicAxes[1] are the indices of the
 * isotropic coupled axes and _2DAnisotropicAxis is the index of the anisotropic
 * axis
 *
 * @return tensor3D
 */
tensor3D SemiIsotropicBerendsenManostat::calculateMu() const
{
    const auto p_xyz = diagonal(_pressureTensor);
    const auto p_x   = p_xyz[_2DIsotropicAxes[0]];
    const auto p_y   = p_xyz[_2DIsotropicAxes[1]];
    const auto p_xy  = (p_x + p_y) / 2.0;
    const auto p_z   = p_xyz[_2DAnisotropicAxis];

    const auto preFactor = _compressibility * _dt / _tau;

    const double mu_xy = ::sqrt(1.0 - preFactor * (_targetPressure - p_xy));
    const double mu_z  = 1.0 - preFactor * (_targetPressure - p_z);

    linearAlgebra::Vec3D mu;

    mu[_2DIsotropicAxes[0]] = mu_xy;
    mu[_2DIsotropicAxes[1]] = mu_xy;
    mu[_2DAnisotropicAxis]  = mu_z;

    return diagonalMatrix(mu);
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (anisotropic)
 *
 * @details If a fixed axis is specified, that axis is not scaled (mu = 1.0)
 * and the other axes are scaled independently
 *
 * @return tensor3D
 */
tensor3D AnisotropicBerendsenManostat::calculateMu() const
{
    using enum FixedAxis;

    const auto pxyz      = diagonal(_pressureTensor);
    const auto preFactor = _compressibility * _dt / _tau;

    auto mu = 1.0 - preFactor * (_targetPressure - pxyz);

    // 2D pressure coupling
    if (_fixedAxis != NONE)
    {
        const auto fixedAxisIndex = static_cast<size_t>(_fixedAxis) - 1;
        mu[fixedAxisIndex]        = 1.0;
    }

    return diagonalMatrix(mu);
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (full
 * anisotropic including angles)
 *
 * @details If a fixed axis is specified, the corresponding row and column
 * are zeroed (no coupling with other axes) and the diagonal is set to 1.0
 *
 * @return tensor3D
 */
tensor3D FullAnisotropicBerendsenManostat::calculateMu() const
{
    using enum FixedAxis;

    const auto pTarget   = diagonalMatrix(_targetPressure);
    const auto preFactor = _compressibility * _dt / _tau;
    const auto kronecker = kroneckerDeltaMatrix<double>();

    auto mu = kronecker - preFactor * (pTarget - _pressureTensor);

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

/**
 * @brief get the isotropy
 *
 * @return Isotropy
 */
Isotropy BerendsenManostat::getIsotropy() const { return Isotropy::ISOTROPIC; }

/**
 * @brief get the isotropy
 *
 * @return Isotropy
 */
Isotropy SemiIsotropicBerendsenManostat::getIsotropy() const
{
    return Isotropy::SEMI_ISOTROPIC;
}

/**
 * @brief get the isotropy
 *
 * @return Isotropy
 */
Isotropy AnisotropicBerendsenManostat::getIsotropy() const
{
    return Isotropy::ANISOTROPIC;
}

/**
 * @brief get the isotropy
 *
 * @return Isotropy
 */
Isotropy FullAnisotropicBerendsenManostat::getIsotropy() const
{
    return Isotropy::FULL_ANISOTROPIC;
}
