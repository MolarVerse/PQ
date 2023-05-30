#include "manostat.hpp"
#include "constants.hpp"

using namespace std;

void Manostat::calculatePressure(const Virial &virial, PhysicalData &physicalData)
{
    auto ekinVirial = physicalData.getKineticEnergyMolecularVector();
    auto forceVirial = virial.getVirial();

    _pressureVector[0] = (2.0 * ekinVirial[0] + forceVirial[0]) / physicalData.getVolume() * _PRESSURE_FACTOR_;

    _pressure = (_pressureVector[0] + _pressureVector[1] + _pressureVector[2]) / 3.0;

    physicalData.setPressure(_pressure);
}