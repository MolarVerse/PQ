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
#include <cmath>        // for pow
#include <functional>   // for identity

using manostat::BerendsenManostat;

/**
 * @brief apply Berendsen manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void BerendsenManostat::applyManostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData)
{
    calculatePressure(simBox, physicalData);

    const auto linearScalingFactor = ::pow(
        1.0 - _compressibility * settings::TimingsSettings::getTimeStep() / _tau * (_targetPressure - _pressure), 1.0 / 3.0);

    const auto scalingFactors = linearAlgebra::Vec3D(linearScalingFactor);

    simBox.scaleBox(scalingFactors);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulombRadiusCutOff(customException::ExceptionType::MANOSTATEXCEPTION);

    auto scaleMolecule = [&scalingFactors](auto &molecule) { molecule.scale(scalingFactors); };

    std::ranges::for_each(simBox.getMolecules(), scaleMolecule);
}