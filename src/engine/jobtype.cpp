#include "jobtype.hpp"

#include <iostream>
#include <cmath>

using namespace std;

void MMMD::calculateForces(SimulationBox &simBox, OutputData &outputData)
{
    vector<double> xyz_i(3);
    vector<double> xyz_j(3);
    vector<double> dxyz(3);

    vector<double> box = simBox._box.getBoxDimensions();

    // inter molecular forces
    for (int mol_i = 0; mol_i < simBox.getNumberOfMolecules(); mol_i++)
    {
        auto &molecule_i = simBox._molecules[mol_i];
        int moltype_i = molecule_i.getMoltype();

        for (int mol_j = 0; mol_j < mol_i; mol_j++)
        {
            auto &molecule_j = simBox._molecules[mol_j];
            int moltype_j = molecule_j.getMoltype();

            for (int atom_i = 0; atom_i < molecule_i.getNumberOfAtoms(); atom_i++)
            {
                for (int atom_j = 0; atom_j < molecule_j.getNumberOfAtoms(); atom_j++)
                {
                    molecule_i.getAtomPosition(atom_i, xyz_i);
                    molecule_j.getAtomPosition(atom_j, xyz_j);

                    // const double distanceSquared = simBox._box.calculateDistanceSquared(xyz_i, xyz_j, dxyz);
                    dxyz[0] = xyz_i[0] - xyz_j[0];
                    dxyz[1] = xyz_i[1] - xyz_j[1];
                    dxyz[2] = xyz_i[2] - xyz_j[2];

                    dxyz[0] -= box[0] * round(dxyz[0] / box[0]);
                    dxyz[1] -= box[1] * round(dxyz[1] / box[1]);
                    dxyz[2] -= box[2] * round(dxyz[2] / box[2]);

                    double distanceSquared = dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2];

                    double RcCutOff = simBox.getRcCutOff();

                    if (distanceSquared < RcCutOff * RcCutOff)
                    {
                        const double distance = sqrt(distanceSquared);
                        const int atomType_i = molecule_i.getAtomType(atom_i);
                        const int atomType_j = molecule_j.getAtomType(atom_j);

                        double coulombCoefficient = simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

                        double energy, force;

                        calcCoulomb(coulombCoefficient, simBox.getRcCutOff(), distance, energy, force, simBox.getcEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getcForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                        outputData.addAverageCoulombEnergy(energy);

                        double rncCutOff = simBox.getRncCutOff(moltype_i, moltype_j, atomType_i, atomType_j);

                        if (distance < rncCutOff)
                        {
                            auto &guffCoefficients = simBox.getGuffCoefficients(moltype_i, moltype_j, atomType_i, atomType_j);

                            calcNonCoulomb(guffCoefficients, rncCutOff, distance, energy, force, simBox.getncEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getncForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                            outputData.addAverageNonCoulombEnergy(energy);
                        }
                    }
                }
            }
        }
    }
}

void JobType::calcCoulomb(double coulombCoefficient, double rcCutoff, double distance, double &energy, double &force, double energy_cutoff, double force_cutoff)
{
    energy = coulombCoefficient * (1 / distance) - energy_cutoff - force_cutoff * (distance - rcCutoff);
    force = -coulombCoefficient * (1 / (distance * distance)) - force_cutoff;
}

void JobType::calcNonCoulomb(vector<double> &guffCoefficients, double rncCutoff, double distance, double &energy, double &force, double energy_cutoff, double force_cutoff)
{
    auto c6 = guffCoefficients[0];
    auto n6 = guffCoefficients[1];
    auto c12 = guffCoefficients[2];
    auto n12 = guffCoefficients[3];

    energy = c6 / pow(distance, n6) + c12 / pow(distance, n12) - energy_cutoff - force_cutoff * (rncCutoff - distance);
    force = -n6 * c6 / pow(distance, n6 + 1) - n12 * c12 / pow(distance, n12 + 1) - force_cutoff;
}