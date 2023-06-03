#include "manostat.hpp"
#include "constants.hpp"

using namespace std;

void Manostat::calculatePressure(PhysicalData &physicalData)
{
    const auto ekinVirial = physicalData.getKineticEnergyVirialVector();
    const auto forceVirial = physicalData.getVirial();
    const auto volume = physicalData.getVolume();

    _pressureVector = (2.0 * ekinVirial + forceVirial) / volume * _PRESSURE_FACTOR_;

    _pressure = mean(_pressureVector);

    physicalData.setPressure(_pressure);
}