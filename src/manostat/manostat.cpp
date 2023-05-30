#include "manostat.hpp"

using namespace std;

void Manostat::calculatePressure(Virial &virial, PhysicalData &physicalData)
{
    auto ekinVirial = physicalData.getKineticEnergyMolecularVector();
    auto forceVirial = virial.getVirial();

    physicalData.setPressure(_pressure);
}