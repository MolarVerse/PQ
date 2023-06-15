#include "potential.hpp"

#include <iostream>
#include <cmath>

using namespace std;
using namespace simulationBox;

void PotentialBruteForce::calculateForces(SimulationBox &simBox, PhysicalData &physicalData, CellList &)
{
    const auto box = simBox.getBoxDimensions();
    const double RcCutOff = simBox.getRcCutOff();

    double totalCoulombicEnergy = 0.0;
    double totalNonCoulombicEnergy = 0.0;
    double energy = 0.0;

    // inter molecular forces
    const size_t numberOfMolecules = simBox.getNumberOfMolecules();

    for (size_t mol_i = 0; mol_i < numberOfMolecules; ++mol_i)
    {
        auto &molecule_i = simBox.getMolecule(mol_i);
        const size_t moltype_i = molecule_i.getMoltype();
        const size_t numberOfAtomsinMolecule_i = molecule_i.getNumberOfAtoms();

        for (size_t mol_j = 0; mol_j < mol_i; ++mol_j)
        {
            auto &molecule_j = simBox.getMolecule(mol_j);
            const size_t moltype_j = molecule_j.getMoltype();
            const size_t numberOfAtomsinMolecule_j = molecule_j.getNumberOfAtoms();

            for (size_t atom_i = 0; atom_i < numberOfAtomsinMolecule_i; ++atom_i)
            {
                for (size_t atom_j = 0; atom_j < numberOfAtomsinMolecule_j; ++atom_j)
                {
                    const auto xyz_i = molecule_i.getAtomPosition(atom_i);
                    const auto xyz_j = molecule_j.getAtomPosition(atom_j);

                    auto dxyz = xyz_i - xyz_j;

                    const auto txyz = -box * round(dxyz / box);

                    dxyz += txyz;

                    const double distanceSquared = normSquared(dxyz);

                    if (distanceSquared < RcCutOff * RcCutOff)
                    {
                        const double distance = sqrt(distanceSquared);
                        const size_t atomType_i = molecule_i.getAtomType(atom_i);
                        const size_t atomType_j = molecule_j.getAtomType(atom_j);

                        const double coulombCoefficient = simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

                        auto force = 0.0;

                        _coulombPotential->calcCoulomb(coulombCoefficient, simBox.getRcCutOff(), distance, energy, force, simBox.getcEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getcForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                        totalCoulombicEnergy += energy;

                        const auto rncCutOff = simBox.getRncCutOff(moltype_i, moltype_j, atomType_i, atomType_j);

                        if (distance < rncCutOff)
                        {
                            const auto &guffCoefficients = simBox.getGuffCoefficients(moltype_i, moltype_j, atomType_i, atomType_j);

                            _nonCoulombPotential->calcNonCoulomb(guffCoefficients, rncCutOff, distance, energy, force, simBox.getncEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getncForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                            totalNonCoulombicEnergy += energy;
                        }

                        force /= distance;

                        const auto forcexyz = force * dxyz;

                        const auto shiftForcexyz = forcexyz * txyz;

                        molecule_i.addAtomForces(atom_i, forcexyz);
                        molecule_j.addAtomForces(atom_j, -forcexyz);

                        molecule_i.addAtomShiftForces(atom_i, shiftForcexyz);
                    }
                }
            }
        }
    }

    physicalData.setCoulombEnergy(totalCoulombicEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombicEnergy);
}

// TODO: check if cutoff is smaller than smallest cell size
void PotentialCellList::calculateForces(SimulationBox &simBox, PhysicalData &physicalData, CellList &cellList)
{
    const auto box = simBox.getBoxDimensions();

    Molecule *molecule_i = nullptr;
    Molecule *molecule_j = nullptr;

    double totalCoulombicEnergy = 0.0;
    double totalNonCoulombicEnergy = 0.0;
    double energy = 0.0;

    const double RcCutOff = simBox.getRcCutOff();

    auto guffCoefficients = vector<double>(22);

    for (const auto &cell_i : cellList.getCells())
    {
        const size_t numberOfMoleculesInCell_i = cell_i.getNumberOfMolecules();

        for (size_t mol_i = 0; mol_i < numberOfMoleculesInCell_i; ++mol_i)
        {
            molecule_i = cell_i.getMolecule(mol_i);
            const size_t moltype_i = molecule_i->getMoltype();

            for (size_t mol_j = 0; mol_j < mol_i; ++mol_j)
            {
                molecule_j = cell_i.getMolecule(mol_j);
                const size_t moltype_j = molecule_j->getMoltype();

                for (const size_t atom_i : cell_i.getAtomIndices(mol_i))
                {
                    const auto xyz_i = molecule_i->getAtomPosition(atom_i);
                    for (const size_t atom_j : cell_i.getAtomIndices(mol_j))
                    {
                        const auto xyz_j = molecule_j->getAtomPosition(atom_j);

                        auto dxyz = xyz_i - xyz_j;

                        const auto txyz = -box * round(dxyz / box);

                        dxyz += txyz;

                        const double distanceSquared = normSquared(dxyz);

                        if (distanceSquared < RcCutOff * RcCutOff)
                        {
                            const double distance = sqrt(distanceSquared);
                            const size_t atomType_i = molecule_i->getAtomType(atom_i);
                            const size_t atomType_j = molecule_j->getAtomType(atom_j);

                            const double coulombCoefficient = simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

                            auto force = 0.0;

                            _coulombPotential->calcCoulomb(coulombCoefficient, simBox.getRcCutOff(), distance, energy, force, simBox.getcEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getcForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                            totalCoulombicEnergy += energy;

                            const auto rncCutOff = simBox.getRncCutOff(moltype_i, moltype_j, atomType_i, atomType_j);

                            if (distance < rncCutOff)
                            {
                                guffCoefficients = simBox.getGuffCoefficients(moltype_i, moltype_j, atomType_i, atomType_j);

                                _nonCoulombPotential->calcNonCoulomb(guffCoefficients, rncCutOff, distance, energy, force, simBox.getncEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getncForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                                totalNonCoulombicEnergy += energy;
                            }

                            force /= distance;

                            const auto forcexyz = force * dxyz;

                            const auto shiftForcexyz = forcexyz * txyz;

                            molecule_i->addAtomForces(atom_i, forcexyz);
                            molecule_j->addAtomForces(atom_j, -forcexyz);

                            molecule_i->addAtomShiftForces(atom_i, shiftForcexyz);
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto numberOfMoleculesInCell_i = cell_i.getNumberOfMolecules();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto numberOfMoleculesInCell_j = cell_j->getNumberOfMolecules();

            for (size_t mol_i = 0; mol_i < numberOfMoleculesInCell_i; ++mol_i)
            {
                molecule_i = cell_i.getMolecule(mol_i);
                const size_t moltype_i = molecule_i->getMoltype();

                for (size_t mol_j = 0; mol_j < numberOfMoleculesInCell_j; ++mol_j)
                {
                    molecule_j = cell_j->getMolecule(mol_j);
                    const size_t moltype_j = molecule_j->getMoltype();

                    if (molecule_i == molecule_j)
                        continue;

                    for (const auto atom_i : cell_i.getAtomIndices(mol_i))
                    {
                        const auto xyz_i = molecule_i->getAtomPosition(atom_i);

                        for (const auto atom_j : cell_j->getAtomIndices(mol_j))
                        {
                            const auto xyz_j = molecule_j->getAtomPosition(atom_j);

                            auto dxyz = xyz_i - xyz_j;

                            const auto txyz = -box * round(dxyz / box);

                            dxyz += txyz;

                            const double distanceSquared = normSquared(dxyz);

                            if (distanceSquared < RcCutOff * RcCutOff)
                            {
                                const double distance = sqrt(distanceSquared);
                                const size_t atomType_i = molecule_i->getAtomType(atom_i);
                                const size_t atomType_j = molecule_j->getAtomType(atom_j);

                                const double coulombCoefficient = simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

                                auto force = 0.0;

                                _coulombPotential->calcCoulomb(coulombCoefficient, simBox.getRcCutOff(), distance, energy, force, simBox.getcEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getcForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                                totalCoulombicEnergy += energy;

                                const auto rncCutOff = simBox.getRncCutOff(moltype_i, moltype_j, atomType_i, atomType_j);

                                if (distance < rncCutOff)
                                {
                                    guffCoefficients = simBox.getGuffCoefficients(moltype_i, moltype_j, atomType_i, atomType_j);

                                    _nonCoulombPotential->calcNonCoulomb(guffCoefficients, rncCutOff, distance, energy, force, simBox.getncEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j), simBox.getncForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

                                    totalNonCoulombicEnergy += energy;
                                }

                                force /= distance;

                                const auto forcexyz = force * dxyz;

                                const auto shiftForcexyz = forcexyz * txyz;

                                molecule_i->addAtomForces(atom_i, forcexyz);
                                molecule_j->addAtomForces(atom_j, -forcexyz);

                                molecule_i->addAtomShiftForces(atom_i, shiftForcexyz);
                            }
                        }
                    }
                }
            }
        }
    }

    physicalData.setCoulombEnergy(totalCoulombicEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombicEnergy);
}

void GuffCoulomb::calcCoulomb(const double coulombCoefficient,
                              const double rcCutoff,
                              const double distance,
                              double &energy,
                              double &force,
                              const double energy_cutoff,
                              const double force_cutoff) const
{
    energy = coulombCoefficient * (1 / distance) - energy_cutoff - force_cutoff * (rcCutoff - distance);
    force += coulombCoefficient * (1 / (distance * distance)) - force_cutoff;
}

void GuffLJ::calcNonCoulomb(const vector<double> &guffCoefficients,
                            const double rncCutoff,
                            const double distance,
                            double &energy,
                            double &force,
                            const double energy_cutoff,
                            const double force_cutoff) const
{
    const double c6 = guffCoefficients[0];
    const double c12 = guffCoefficients[2];

    const double distance_6 = distance * distance * distance * distance * distance * distance;
    const double distance_12 = distance_6 * distance_6;

    energy = c12 / distance_12 + c6 / distance_6;
    force += 12 * c12 / (distance_12 * distance) + 6 * c6 / (distance_6 * distance);

    energy += energy_cutoff + force_cutoff * (rncCutoff - distance);
    force += force_cutoff;
}

void GuffBuckingham::calcNonCoulomb(const vector<double> &guffCoefficients,
                                    const double rncCutoff,
                                    const double distance,
                                    double &energy,
                                    double &force,
                                    const double energy_cutoff,
                                    const double force_cutoff) const
{
    const double c1 = guffCoefficients[0];
    const double c2 = guffCoefficients[1];
    const double c3 = guffCoefficients[2];

    const double helper = c1 * exp(distance * c2);

    const double distance_6 = distance * distance * distance * distance * distance * distance;
    const double helper_c3 = c3 / distance_6;

    energy = helper + helper_c3;
    force += -c2 * helper + 6 * helper_c3 / distance;

    energy += energy_cutoff + force_cutoff * (rncCutoff - distance);
    force += force_cutoff;
}

void GuffNonCoulomb::calcNonCoulomb(const vector<double> &guffCoefficients,
                                    const double rncCutoff,
                                    const double distance,
                                    double &energy,
                                    double &force,
                                    const double energy_cutoff,
                                    const double force_cutoff) const
{
    const double c1 = guffCoefficients[0];
    const double n2 = guffCoefficients[1];
    const double c3 = guffCoefficients[2];
    const double n4 = guffCoefficients[3];

    energy = c1 / pow(distance, n2) + c3 / pow(distance, n4);
    force += n2 * c1 / pow(distance, n2 + 1) + n4 * c3 / pow(distance, n4 + 1);

    const double c5 = guffCoefficients[4];
    const double n6 = guffCoefficients[5];
    const double c7 = guffCoefficients[6];
    const double n8 = guffCoefficients[7];

    energy += c5 / pow(distance, n6) + c7 / pow(distance, n8);
    force += n6 * c5 / pow(distance, n6 + 1) + n8 * c7 / pow(distance, n8 + 1);

    const double c9 = guffCoefficients[8];
    const double cexp10 = guffCoefficients[9];
    const double rexp11 = guffCoefficients[10];

    double helper = exp(cexp10 * (distance - rexp11));

    energy += c9 / (1 + helper);
    force += c9 * cexp10 * helper / ((1 + helper) * (1 + helper));

    const double c12 = guffCoefficients[11];
    const double cexp13 = guffCoefficients[12];
    const double rexp14 = guffCoefficients[13];

    helper = exp(cexp13 * (distance - rexp14));

    energy += c12 / (1 + helper);
    force += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));

    const double c15 = guffCoefficients[14];
    const double cexp16 = guffCoefficients[15];
    const double rexp17 = guffCoefficients[16];
    const double n18 = guffCoefficients[17];

    helper = c15 * exp(cexp16 * pow((distance - rexp17), n18));

    energy += helper;
    force += -cexp16 * n18 * pow((distance - rexp17), n18 - 1) * helper;

    const double c19 = guffCoefficients[18];
    const double cexp20 = guffCoefficients[19];
    const double rexp21 = guffCoefficients[20];
    const double n22 = guffCoefficients[21];

    helper = c19 * exp(cexp20 * pow((distance - rexp21), n22));

    energy += helper;
    force += -cexp20 * n22 * pow((distance - rexp21), n22 - 1) * helper;

    energy += -energy_cutoff - force_cutoff * (rncCutoff - distance);
    force += -force_cutoff;
}