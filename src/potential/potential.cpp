#include "potential.hpp"

#include <cmath>
#include <iostream>

using namespace std;
using namespace simulationBox;
using namespace potential;
using namespace physicalData;
using namespace linearAlgebra;

std::pair<double, double> Potential::calculateSingleInteraction(const linearAlgebra::Vec3D &box,
                                                                simulationBox::Molecule    &molecule1,
                                                                simulationBox::Molecule    &molecule2,
                                                                const size_t                atom1,
                                                                const size_t                atom2)
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto xyz_i = molecule1.getAtomPosition(atom1);
    const auto xyz_j = molecule2.getAtomPosition(atom2);

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -box * round(dxyz / box);

    // dxyz += txyz;
    dxyz[0] += txyz[0];
    dxyz[1] += txyz[1];
    dxyz[2] += txyz[2];

    const double distanceSquared = normSquared(dxyz);

    if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff(); distanceSquared < RcCutOff * RcCutOff)
    {
        const double distance   = ::sqrt(distanceSquared);
        const size_t atomType_i = molecule1.getAtomType(atom1);
        const size_t atomType_j = molecule2.getAtomType(atom2);

        // const size_t externalGlobalVdwType_i = molecule1.getExternalGlobalVDWType(atom1);
        // const size_t externalGlobalVdwType_j = molecule2.getExternalGlobalVDWType(atom2);

        // const size_t globalVdwType_i =
        // simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_i); const size_t globalVdwType_j
        // = simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_j);

        const size_t globalVdwType_i = 0;
        const size_t globalVdwType_j = 0;

        const auto moltype_i = molecule1.getMoltype();
        const auto moltype_j = molecule2.getMoltype();

        const auto combinedIndices = {moltype_i, moltype_j, atomType_i, atomType_j, globalVdwType_i, globalVdwType_j};

        const auto coulombPreFactor = 1.0;   // TODO: implement for force field

        auto [energy, force] = _coulombPotential->calculate(combinedIndices, distance, coulombPreFactor);
        coulombEnergy        = energy;

        const auto nonCoulombicPair = _nonCoulombPotential->getNonCoulombPair(combinedIndices);

        if (const auto rncCutOff = nonCoulombicPair->getRadialCutOff(); distance < rncCutOff)
        {
            const auto &[energy, nonCoulombForce] = nonCoulombicPair->calculateEnergyAndForce(distance);
            nonCoulombEnergy                      = energy;

            force += nonCoulombForce;
        }

        force /= distance;

        const auto forcexyz = force * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        molecule1.addAtomForce(atom1, forcexyz);
        molecule2.addAtomForce(atom2, -forcexyz);

        molecule1.addAtomShiftForce(atom1, shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}

/**
 * @brief calculates forces, coulombic and non-coulombic energy for brute force routine
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialBruteForce::calculateForces(SimulationBox &simBox, PhysicalData &physicalData, CellList &)
{
    const auto box = simBox.getBoxDimensions();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // inter molecular forces
    const size_t numberOfMolecules = simBox.getNumberOfMolecules();

    for (size_t mol1 = 0; mol1 < numberOfMolecules; ++mol1)
    {
        auto        &molecule1                 = simBox.getMolecule(mol1);
        const size_t numberOfAtomsInMolecule_i = molecule1.getNumberOfAtoms();

        for (size_t mol2 = 0; mol2 < mol1; ++mol2)
        {
            auto        &molecule2                 = simBox.getMolecule(mol2);
            const size_t numberOfAtomsInMolecule_j = molecule2.getNumberOfAtoms();

            for (size_t atom1 = 0; atom1 < numberOfAtomsInMolecule_i; ++atom1)
            {
                for (size_t atom2 = 0; atom2 < numberOfAtomsInMolecule_j; ++atom2)
                {
                    const auto [coulombEnergy, nonCoulombEnergy] =
                        calculateSingleInteraction(box, molecule1, molecule2, atom1, atom2);

                    totalCoulombEnergy    += coulombEnergy;
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }
            }
        }
    }

    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);
}

// TODO: check if cutoff is smaller than smallest cell size
/**
 * @brief calculates forces, coulombic and non-coulombic energy for cell list routine
 *
 * @param simBox
 * @param physicalData
 * @param cellList
 */
inline void PotentialCellList::calculateForces(SimulationBox &simBox, PhysicalData &physicalData, CellList &cellList)
{
    const auto box = simBox.getBoxDimensions();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    for (const auto &cell1 : cellList.getCells())
    {
        const auto numberOfMolecules = cell1.getNumberOfMolecules();

        for (size_t mol1 = 0; mol1 < numberOfMolecules; ++mol1)
        {
            auto *molecule1 = cell1.getMolecule(mol1);

            for (size_t mol2 = 0; mol2 < mol1; ++mol2)
            {
                auto *molecule2 = cell1.getMolecule(mol2);

                for (const size_t atom1 : cell1.getAtomIndices(mol1))
                {
                    for (const size_t atom2 : cell1.getAtomIndices(mol2))
                    {
                        const auto [coulombEnergy, nonCoulombEnergy] =
                            calculateSingleInteraction(box, *molecule1, *molecule2, atom1, atom2);

                        totalCoulombEnergy    += coulombEnergy;
                        totalNonCoulombEnergy += nonCoulombEnergy;
                    }
                }
            }
        }
    }
    for (const auto &cell1 : cellList.getCells())
    {
        const auto numberOfMoleculesInCell_i = cell1.getNumberOfMolecules();

        for (const auto *cell2 : cell1.getNeighbourCells())
        {
            const auto numberOfMoleculesInCell_j = cell2->getNumberOfMolecules();

            for (size_t mol1 = 0; mol1 < numberOfMoleculesInCell_i; ++mol1)
            {
                auto *molecule1 = cell1.getMolecule(mol1);

                for (const auto atom1 : cell1.getAtomIndices(mol1))
                {
                    for (size_t mol2 = 0; mol2 < numberOfMoleculesInCell_j; ++mol2)
                    {
                        auto *molecule2 = cell2->getMolecule(mol2);

                        if (molecule1 == molecule2)
                            continue;

                        for (const auto atom2 : cell2->getAtomIndices(mol2))
                        {
                            const auto [coulombEnergy, nonCoulombEnergy] =
                                calculateSingleInteraction(box, *molecule1, *molecule2, atom1, atom2);

                            totalCoulombEnergy    += coulombEnergy;
                            totalNonCoulombEnergy += nonCoulombEnergy;
                        }
                    }
                }
            }
        }
    }

    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);
}