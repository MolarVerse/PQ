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
        const auto &molecule_i = simBox._molecules[mol_i];
        int moltype_i = molecule_i.getMoltype();

        for (int mol_j = 0; mol_j < mol_i; mol_j++)
        {
            const auto &molecule_j = simBox._molecules[mol_j];
            int moltype_j = molecule_j.getMoltype();

            for (int atom_i = 0; atom_i < molecule_i.getNumberOfAtoms(); atom_i++)
            {
                for (int atom_j = 0; atom_j < molecule_j.getNumberOfAtoms(); atom_j++)
                {
                    molecule_i.getAtomPosition(atom_i, xyz_i);
                    molecule_j.getAtomPosition(atom_j, xyz_j);

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

                        double energy = 0.0;
                        double force = 0.0;

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

void MMMD::calculateForcesCellList(SimulationBox &simBox, OutputData &outputData, CellList &cellList)
{
    vector<double> xyz_i(3);
    vector<double> xyz_j(3);
    vector<double> dxyz(3);

    vector<double> box = simBox._box.getBoxDimensions();

    const Molecule *molecule_i;
    const Molecule *molecule_j;

    for (const auto &cell_i : cellList.getCells())
    {
        for (int mol_i = 0; mol_i < cell_i.getNumberOfMolecules(); mol_i++)
        {
            molecule_i = cell_i.getMolecule(mol_i);
            int moltype_i = molecule_i->getMoltype();

            for (int mol_j = 0; mol_j < mol_i; mol_j++)
            {
                molecule_j = cell_i.getMolecule(mol_j);
                int moltype_j = molecule_j->getMoltype();

                for (int atom_i = 0; atom_i < molecule_i->getNumberOfAtoms(); atom_i++)
                {
                    for (int atom_j = 0; atom_j < molecule_j->getNumberOfAtoms(); atom_j++)
                    {
                        molecule_i->getAtomPosition(atom_i, xyz_i);
                        molecule_j->getAtomPosition(atom_j, xyz_j);

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
                            const int atomType_i = molecule_i->getAtomType(atom_i);
                            const int atomType_j = molecule_j->getAtomType(atom_j);

                            double coulombCoefficient = simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

                            double energy = 0.0;
                            double force = 0.0;

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

    for (const auto &cell_i : cellList.getCells())
    {
        for (const auto cell_j : cell_i.getNeighbourCells())
        {
            for (int mol_i = 0; mol_i < cell_i.getNumberOfMolecules(); mol_i++)
            {
                molecule_i = cell_i.getMolecule(mol_i);
                int moltype_i = molecule_i->getMoltype();

                for (int mol_j = 0; mol_j < cell_j->getNumberOfMolecules(); mol_j++)
                {
                    molecule_j = cell_j->getMolecule(mol_j);
                    int moltype_j = molecule_j->getMoltype();

                    for (int atom_i = 0; atom_i < molecule_i->getNumberOfAtoms(); atom_i++)
                    {
                        for (int atom_j = 0; atom_j < molecule_j->getNumberOfAtoms(); atom_j++)
                        {
                            molecule_i->getAtomPosition(atom_i, xyz_i);
                            molecule_j->getAtomPosition(atom_j, xyz_j);

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
                                const int atomType_i = molecule_i->getAtomType(atom_i);
                                const int atomType_j = molecule_j->getAtomType(atom_j);

                                double coulombCoefficient = simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

                                double energy = 0.0;
                                double force = 0.0;

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
    }
}

void JobType::calcCoulomb(double coulombCoefficient, double rcCutoff, double distance, double &energy, double &force, double energy_cutoff, double force_cutoff) const
{
    energy = coulombCoefficient * (1 / distance) - energy_cutoff - force_cutoff * (rcCutoff - distance);
    force = coulombCoefficient * (1 / (distance * distance)) - force_cutoff;
}

void JobType::calcLJ(vector<double> &guffCoefficients, double rncCutoff, double distance, double &energy, double &force, double energy_cutoff, double force_cutoff) const
{
    auto c6 = guffCoefficients[0];
    auto c12 = guffCoefficients[2];

    auto distance_6 = distance * distance * distance * distance * distance * distance;
    auto distance_12 = distance_6 * distance_6;

    energy = c12 / distance_12 + c6 / distance_6;
    force = 12 * c12 / (distance_12 * distance) + 6 * c6 / (distance_6 * distance);

    energy += energy_cutoff + force_cutoff * (rncCutoff - distance);
    force += force_cutoff;
}

void JobType::calcBuckingham(vector<double> &guffCoefficients, double rncCutoff, double distance, double &energy, double &force, double energy_cutoff, double force_cutoff) const
{
    auto c1 = guffCoefficients[0];
    auto c2 = guffCoefficients[1];
    auto c3 = guffCoefficients[2];

    auto helper = c1 * exp(distance * c2);

    auto distance_6 = distance * distance * distance * distance * distance * distance;
    auto helper_c3 = c3 / distance_6;

    energy = helper + helper_c3;
    force = -c2 * helper + 6 * helper_c3 / distance;

    energy += energy_cutoff + force_cutoff * (rncCutoff - distance);
    force += force_cutoff;
}

void JobType::calcNonCoulomb(vector<double> &guffCoefficients, double rncCutoff, double distance, double &energy, double &force, double energy_cutoff, double force_cutoff) const
{
    auto c1 = guffCoefficients[0];
    auto n2 = guffCoefficients[1];
    auto c3 = guffCoefficients[2];
    auto n4 = guffCoefficients[3];

    energy = c1 / pow(distance, n2) + c3 / pow(distance, n4);
    force = n2 * c1 / pow(distance, n2 + 1) + n4 * c3 / pow(distance, n4 + 1);

    auto c5 = guffCoefficients[4];
    auto n6 = guffCoefficients[5];
    auto c7 = guffCoefficients[6];
    auto n8 = guffCoefficients[7];

    energy += c5 / pow(distance, n6) + c7 / pow(distance, n8);
    force += n6 * c5 / pow(distance, n6 + 1) + n8 * c7 / pow(distance, n8 + 1);

    auto c9 = guffCoefficients[8];
    auto cexp10 = guffCoefficients[9];
    auto rexp11 = guffCoefficients[10];

    auto helper = exp(cexp10 * (distance - rexp11));

    energy += c9 / (1 + helper);
    force += c9 * cexp10 * helper / ((1 + helper) * (1 + helper));

    auto c12 = guffCoefficients[11];
    auto cexp13 = guffCoefficients[12];
    auto rexp14 = guffCoefficients[13];

    helper = exp(cexp13 * (distance - rexp14));

    energy += c12 / (1 + helper);
    force += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));

    auto c15 = guffCoefficients[14];
    auto cexp16 = guffCoefficients[15];
    auto rexp17 = guffCoefficients[16];
    auto n18 = guffCoefficients[17];

    helper = c15 * exp(cexp16 * pow((distance - rexp17), n18));

    energy += helper;
    force += -cexp16 * n18 * pow((distance - rexp17), n18 - 1) * helper;

    auto c19 = guffCoefficients[18];
    auto cexp20 = guffCoefficients[19];
    auto rexp21 = guffCoefficients[20];
    auto n22 = guffCoefficients[21];

    helper = c19 * exp(cexp20 * pow((distance - rexp21), n22));

    energy += helper;
    force += -cexp20 * n22 * pow((distance - rexp21), n22 - 1) * helper;

    energy += -energy_cutoff - force_cutoff * (rncCutoff - distance);
    force += -force_cutoff;
}