#include "manostat.hpp"

#include "constants.hpp"

using namespace std;
using namespace simulationBox;

void Manostat::calculatePressure(PhysicalData &physicalData)
{
    const auto ekinVirial  = physicalData.getKineticEnergyVirialVector();
    const auto forceVirial = physicalData.getVirial();
    const auto volume      = physicalData.getVolume();

    _pressureVector = (2.0 * ekinVirial + forceVirial) / volume * _PRESSURE_FACTOR_;

    _pressure = mean(_pressureVector);

    physicalData.setPressure(_pressure);
}

void Manostat::applyManostat(SimulationBox &, PhysicalData &physicalData) { calculatePressure(physicalData); }

void BerendsenManostat::applyManostat(SimulationBox &simBox, PhysicalData &physicalData)
{
    calculatePressure(physicalData);

    const auto scaleFactors = Vec3D(pow(1.0 - _compressability * _timestep / _tau * (_targetPressure - _pressure), 1.0 / 3.0));

    simBox.scaleBox(scaleFactors);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    for (auto &molecule : simBox.getMolecules())
        molecule.scale(scaleFactors);

    // calculatePressure(physicalData); TODO: talk to thh about this
}