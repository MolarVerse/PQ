#include "virial.hpp"

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

void Virial::calculateVirial(SimulationBox &simulationBox, PhysicalData &physicalData)
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

void VirialMolecular::calculateVirial(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    Virial::calculateVirial(simulationBox, physicalData);

    cout << "Virial: " << _virial[0] << " " << _virial[1] << " " << _virial[2] << endl;

    intraMolecularVirialCorrection(simulationBox);

    cout << "Virial: " << _virial[0] << " " << _virial[1] << " " << _virial[2] << endl;

    physicalData.setVirial(_virial);
}

void VirialMolecular::intraMolecularVirialCorrection(SimulationBox &simulationBox)
{
    vector<double> xyz(3);
    vector<double> dxyz(3);
    vector<double> centerOfMass(3);
    vector<double> forcexyz(3);

    vector<double> box = simulationBox._box.getBoxDimensions();

    for (const auto &molecule : simulationBox._molecules)
    {
        molecule.getCenterOfMass(centerOfMass);

        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            molecule.getAtomForces(i, forcexyz);
            molecule.getAtomPositions(i, xyz);

            dxyz[0] = xyz[0] - centerOfMass[0];
            dxyz[1] = xyz[1] - centerOfMass[1];
            dxyz[2] = xyz[2] - centerOfMass[2];

            dxyz[0] -= box[0] * round(dxyz[0] / box[0]);
            dxyz[1] -= box[1] * round(dxyz[1] / box[1]);
            dxyz[2] -= box[2] * round(dxyz[2] / box[2]);

            // cout << _virial[0] << " " << _virial[1] << " " << _virial[2] << endl;

            _virial[0] -= forcexyz[0] * dxyz[0];
            _virial[1] -= forcexyz[1] * dxyz[1];
            _virial[2] -= forcexyz[2] * dxyz[2];

            // cout << _virial[0] << " " << _virial[1] << " " << _virial[2] << endl;
            // exit(-1);
        }
    }
}