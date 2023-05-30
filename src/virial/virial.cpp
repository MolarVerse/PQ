#include "virial.hpp"

#include <vector>

using namespace std;

void VirialMolecular::computeVirial(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    _virial = {0.0, 0.0, 0.0};

    vector<double> forcexyz(3);
    vector<double> shiftForcexyz(3);
    vector<double> xyz(3);

    for (auto &molecule : simulationBox._molecules)
    {
        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            molecule.getAtomForces(i, forcexyz);
            molecule.getAtomShiftForces(i, shiftForcexyz);
            molecule.getAtomPositions(i, xyz);

            _virial[0] += forcexyz[0] * xyz[0] + shiftForcexyz[0];
            _virial[1] += forcexyz[1] * xyz[1] + shiftForcexyz[1];
            _virial[2] += forcexyz[2] * xyz[2] + shiftForcexyz[2];

            molecule.setAtomShiftForces(i, {0.0, 0.0, 0.0});
        }
    }

    physicalData.setVirial(_virial);
}