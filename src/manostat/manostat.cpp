#include "manostat.hpp"

#include "constants.hpp"       // for _PRESSURE_FACTOR_
#include "exceptions.hpp"      // for ExceptionType
#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData, physicalData, simulationBox
#include "simulationBox.hpp"   // for SimulationBox

#include <cmath>        // for pow
#include <functional>   // for function
#include <vector>       // for vector

using namespace simulationBox;
using namespace physicalData;
using namespace manostat;
using namespace linearAlgebra;

/**
 * @brief calculate the pressure of the system
 *
 * @param physicalData
 */
void Manostat::calculatePressure(PhysicalData &physicalData)
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
void Manostat::applyManostat(SimulationBox &, PhysicalData &physicalData) { calculatePressure(physicalData); }

/**
 * @brief apply Berendsen manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void BerendsenManostat::applyManostat(SimulationBox &simBox, PhysicalData &physicalData)
{
    calculatePressure(physicalData);

    const auto scaleFactors = Vec3D(::pow(1.0 - _compressibility * _timestep / _tau * (_targetPressure - _pressure), 1.0 / 3.0));

    simBox.scaleBox(scaleFactors);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulombRadiusCutOff(customException::ExceptionType::MANOSTATEXCEPTION);

    for (auto &molecule : simBox.getMolecules())
        molecule.scale(scaleFactors);
}