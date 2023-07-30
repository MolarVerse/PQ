#include "manostat.hpp"

#include "constants.hpp"

using namespace std;
using namespace simulationBox;
using namespace physicalData;
using namespace manostat;
using namespace vector3d;

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

    for (auto &molecule : simBox.getMolecules())
        molecule.scale(scaleFactors);

    // calculatePressure(physicalData); TODO: talk to thh about this
}