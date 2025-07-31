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

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for cbrt
#include <functional>   // for identity

#include "exceptions.hpp"        // for ExceptionType
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "staticMatrix.hpp"      // for diagonal, diagonalMatrix, trace
#include "timingsSettings.hpp"   // for TimingsSettings
#include "vector3d.hpp"          // for Vec3D

using namespace linearAlgebra;
using namespace settings;
using namespace manostat;
using namespace customException;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief Construct a new Berendsen Manostat:: Berendsen Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 */
BerendsenManostat::BerendsenManostat(
    const double targetPressure,
    const double tau,
    const double compressibility
)
    : Manostat(targetPressure), _tau(tau), _compressibility(compressibility)
{
    _dt = TimingsSettings::getTimeStep();
}

/**
 * @brief Construct a new Berendsen Manostat:: Berendsen Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 */
SemiIsotropicBerendsenManostat::SemiIsotropicBerendsenManostat(
    const double               targetPressure,
    const double               tau,
    const double               compressibility,
    const size_t               anisotropicAxis,
    const std::vector<size_t> &isotropicAxes
)
    : BerendsenManostat(targetPressure, tau, compressibility),
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
    startTimingsSection("Berendsen");

    calculatePressure(simBox, physicalData);

    const auto mu = calculateMu();

    simBox.scaleBox(mu);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulRadiusCutOff(ExceptionType::MANOSTATEXCEPTION);

    auto scaleMolecule = [&mu, &simBox](auto &molecule)
    { molecule.scale(mu, simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), scaleMolecule);

    stopTimingsSection("Berendsen");
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (isotropic)
 *
 * @return tensor3D
 */
tensor3D BerendsenManostat::calculateMu() const
{
    const auto p         = trace(_pressureTensor) / 3.0;
    const auto preFactor = _compressibility * _dt / _tau;

    return diagonalMatrix(::cbrt(1.0 - preFactor * (_targetPressure - p)));
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
 * @return tensor3D
 */
tensor3D AnisotropicBerendsenManostat::calculateMu() const
{
    const auto pxyz      = diagonal(_pressureTensor);
    const auto preFactor = _compressibility * _dt / _tau;

    return diagonalMatrix(1.0 - preFactor * (_targetPressure - pxyz));
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (full
 * anisotropic including angles)
 *
 * @return tensor3D
 */
tensor3D FullAnisotropicBerendsenManostat::calculateMu() const
{
    const auto pTarget   = diagonalMatrix(_targetPressure);
    const auto preFactor = _compressibility * _dt / _tau;
    const auto kronecker = kroneckerDeltaMatrix<double>();

    auto mu = kronecker - preFactor * (pTarget - _pressureTensor);

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