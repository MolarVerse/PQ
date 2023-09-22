#include "berendsenManostat.hpp"

#include "constants.hpp"         // for _PRESSURE_FACTOR_
#include "exceptions.hpp"        // for ExceptionType
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings

#include <cmath>   // for pow

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