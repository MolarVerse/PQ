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

#include "berendsenManostat.hpp"

#include "exceptions.hpp"        // for ExceptionType
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings
#include "vector3d.hpp"          // for Vec3D

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for cbrt
#include <functional>   // for identity

using manostat::AnisotropicBerendsenManostat;
using manostat::BerendsenManostat;
using manostat::SemiIsotropicBerendsenManostat;

/**
 * @brief Construct a new Berendsen Manostat:: Berendsen Manostat object
 *
 * @param targetPressure
 * @param tau
 * @param compressibility
 */
BerendsenManostat::BerendsenManostat(const double targetPressure, const double tau, const double compressibility)
    : Manostat(targetPressure), _tau(tau), _compressibility(compressibility)
{
    _dt = settings::TimingsSettings::getTimeStep();
}

/**
 * @brief apply Berendsen manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void BerendsenManostat::applyManostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData)
{
    calculatePressure(simBox, physicalData);

    const auto mu = calculateMu();

    simBox.scaleBox(mu);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulombRadiusCutOff(customException::ExceptionType::MANOSTATEXCEPTION);

    auto scaleMolecule = [&mu, &simBox](auto &molecule) { molecule.scale(mu, simBox.getBox()); };

    std::ranges::for_each(simBox.getMolecules(), scaleMolecule);
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (isotropic)
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D BerendsenManostat::calculateMu() const
{
    return linearAlgebra::Vec3D(::cbrt(1.0 - _compressibility * _dt / _tau * (_targetPressure - trace(_pressureTensor) / 3.0)));
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (semi-isotropic)
 *
 * @details _2DIsotropicAxes[0] and _2DIsotropicAxes[1] are the indices of the isotropic coupled axes and _2DAnisotropicAxis is
 * the index of the anisotropic axis
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D SemiIsotropicBerendsenManostat::calculateMu() const
{
    const auto p_xyz = diagonal(_pressureTensor);
    const auto p_x   = p_xyz[_2DIsotropicAxes[0]];
    const auto p_y   = p_xyz[_2DIsotropicAxes[1]];
    const auto p_xy  = (p_x + p_y) / 2.0;
    const auto p_z   = p_xyz[_2DAnisotropicAxis];

    const double mu_xy = ::sqrt(1.0 - _compressibility * _dt / _tau * (_targetPressure - p_xy));
    const double mu_z  = 1.0 - _compressibility * _dt / _tau * (_targetPressure - p_z);

    linearAlgebra::Vec3D mu;

    mu[_2DIsotropicAxes[0]] = mu_xy;
    mu[_2DIsotropicAxes[1]] = mu_xy;
    mu[_2DAnisotropicAxis]  = mu_z;

    return mu;
}

/**
 * @brief calculate mu as scaling factor for Berendsen manostat (anisotropic)
 *
 * @return linearAlgebra::Vec3D
 */
linearAlgebra::Vec3D AnisotropicBerendsenManostat::calculateMu() const
{
    return linearAlgebra::Vec3D(1.0 - _compressibility * _dt / _tau * (_targetPressure - diagonal(_pressureTensor)));
}