#include "manostat.hpp"

#include "constants.hpp"       // for _PRESSURE_FACTOR_
#include "exceptions.hpp"      // for ExceptionType
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for pow
#include <functional>   // for identity, function

using namespace manostat;

/**
 * @brief calculate the pressure of the system
 *
 * @param physicalData
 */
void Manostat::calculatePressure(physicalData::PhysicalData &physicalData)
{
    const auto ekinVirial  = physicalData.getKineticEnergyVirialVector();
    const auto forceVirial = physicalData.getVirial();
    const auto volume      = physicalData.getVolume();

    _pressureVector = (2.0 * ekinVirial + forceVirial) / volume * constants::_PRESSURE_FACTOR_;

    _pressure = mean(_pressureVector);

    physicalData.setPressure(_pressure);
}

/**
 * @brief apply dummy manostat for NVT ensemble
 *
 * @param physicalData
 */
void Manostat::applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &physicalData)
{
    calculatePressure(physicalData);
}

/**
 * @brief apply Berendsen manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void BerendsenManostat::applyManostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData)
{
    calculatePressure(physicalData);

    const auto linearScalingFactor = ::pow(1.0 - _compressibility * _timestep / _tau * (_targetPressure - _pressure), 1.0 / 3.0);
    const auto scalingFactors      = linearAlgebra::Vec3D(linearScalingFactor);

    simBox.scaleBox(scalingFactors);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulombRadiusCutOff(customException::ExceptionType::MANOSTATEXCEPTION);

    auto scaleMolecule = [&scalingFactors](auto &molecule) { molecule.scale(scalingFactors); };

    std::ranges::for_each(simBox.getMolecules(), scaleMolecule);
}