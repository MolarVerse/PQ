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