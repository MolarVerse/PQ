#include "manostat.hpp"
#include "constants.hpp"

using namespace std;

void Manostat::calculatePressure(const Virial &virial, PhysicalData &physicalData)
{
    auto ekinVirial = physicalData.getKineticEnergyVirialVector();
    auto forceVirial = virial.getVirial();

    _pressureVector[0] = (2.0 * ekinVirial[0] + forceVirial[0]) / physicalData.getVolume() * _PRESSURE_FACTOR_;
    _pressureVector[1] = (2.0 * ekinVirial[1] + forceVirial[1]) / physicalData.getVolume() * _PRESSURE_FACTOR_;
    _pressureVector[2] = (2.0 * ekinVirial[2] + forceVirial[2]) / physicalData.getVolume() * _PRESSURE_FACTOR_;

    _pressure = (_pressureVector[0] + _pressureVector[1] + _pressureVector[2]) / 3.0;

    physicalData.setPressure(_pressure);
}