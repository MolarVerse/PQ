#include "jobtype.hpp"

#include <iostream>
#include <cmath>

using namespace std;

void calcCoulomb(double rcCutoff, double coulombCoefficient, double distance, double &energy, double &force);
void calcNonCoulomb(double rncCutoff, vector<double> &guffCoefficients, double distance, double &energy, double &force);

void MMMD::calculateForces(SimulationBox &simBox, OutputData &outputData)
{
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
                    const auto &xyz_i = molecule_i.getAtomPosition(atom_i);
                    const auto &xyz_j = molecule_j.getAtomPosition(atom_j);

                    const double distance = simBox._box.calculateDistance(xyz_i, xyz_j);

                    if (distance < simBox.getRcCutOff())
                    {
                        const int atomType_i = molecule_i.getAtomType(atom_i);
                        const int atomType_j = molecule_j.getAtomType(atom_j);

                        double coulombCoefficient = simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

                        double energy, force;

                        calcCoulomb(simBox.getRcCutOff(), coulombCoefficient, distance, energy, force);

                        outputData.addAverageCoulombEnergy(energy);

                        double rncCutOff = simBox.getRncCutOff(moltype_i, moltype_j, atomType_i, atomType_j);

                        if (distance < rncCutOff)
                        {
                            auto &guffCoefficients = simBox.getGuffCoefficients(moltype_i, moltype_j, atomType_i, atomType_j);

                            calcNonCoulomb(rncCutOff, guffCoefficients, distance, energy, force);

                            outputData.addAverageNonCoulombEnergy(energy);
                        }
                    }
                }
            }
        }
    }
}

void calcCoulomb(double rcCutoff, double coulombCoefficient, double distance, double &energy, double &force)
{
    energy = coulombCoefficient * (1 / distance - 1 / rcCutoff - 1 / (rcCutoff * rcCutoff) * (rcCutoff - distance));
    force = coulombCoefficient * (1 / (rcCutoff * rcCutoff) - 1 / (distance * distance));
}

void calcNonCoulomb(double rncCutoff, vector<double> &guffCoefficients, double distance, double &energy, double &force)
{
    auto c6 = guffCoefficients[0];
    auto c12 = guffCoefficients[2];

    double force_cutoff = -c6 / pow(rncCutoff, 7) + 6 * c12 / pow(rncCutoff, 13);
    double energy_cutoff = c6 / pow(rncCutoff, 6) - c12 / pow(rncCutoff, 12);
    energy = c6 / pow(distance, 6) - c12 / pow(distance, 12) - energy_cutoff - force_cutoff * (rncCutoff - distance);
}