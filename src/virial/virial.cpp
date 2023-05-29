#include "virial.hpp"

#include <vector>

using namespace std;

void VirialMolecular::computeVirial(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    _virial = {0.0, 0.0, 0.0};

    vector<double> fxyz(3, 0.0);

    // for (auto &molecule : simulationBox._molecules)
    // {
    //     for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
    //     {
    //         molecule.getAtomForce(i, fxyz);
    //     }
    // }
}