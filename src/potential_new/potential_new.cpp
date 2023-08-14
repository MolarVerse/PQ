#include "potential_new.hpp"

#include <cmath>
#include <iostream>

using namespace std;
using namespace simulationBox;
using namespace potential_new;
using namespace physicalData;
using namespace linearAlgebra;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for brute force routine
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialBruteForce::calculateForces(SimulationBox &simBox, PhysicalData &physicalData, CellList &)
{
    const auto   box      = simBox.getBoxDimensions();
    const double RcCutOff = simBox.getCoulombRadiusCutOff();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // inter molecular forces
    const size_t numberOfMolecules = simBox.getNumberOfMolecules();

    for (size_t mol_i = 0; mol_i < numberOfMolecules; ++mol_i)
    {
        auto        &molecule_i                = simBox.getMolecule(mol_i);
        const size_t moltype_i                 = molecule_i.getMoltype();
        const size_t numberOfAtomsInMolecule_i = molecule_i.getNumberOfAtoms();

        for (size_t mol_j = 0; mol_j < mol_i; ++mol_j)
        {
            auto        &molecule_j                = simBox.getMolecule(mol_j);
            const size_t moltype_j                 = molecule_j.getMoltype();
            const size_t numberOfAtomsInMolecule_j = molecule_j.getNumberOfAtoms();

            for (size_t atom_i = 0; atom_i < numberOfAtomsInMolecule_i; ++atom_i)
            {
                for (size_t atom_j = 0; atom_j < numberOfAtomsInMolecule_j; ++atom_j)
                {
                    const auto xyz_i = molecule_i.getAtomPosition(atom_i);
                    const auto xyz_j = molecule_j.getAtomPosition(atom_j);

                    auto dxyz = xyz_i - xyz_j;

                    const auto txyz = -box * round(dxyz / box);

                    // dxyz += txyz;
                    dxyz[0] += txyz[0];
                    dxyz[1] += txyz[1];
                    dxyz[2] += txyz[2];

                    const double distanceSquared = normSquared(dxyz);

                    if (distanceSquared < RcCutOff * RcCutOff)
                    {
                        const double distance   = ::sqrt(distanceSquared);
                        const size_t atomType_i = molecule_i.getAtomType(atom_i);
                        const size_t atomType_j = molecule_j.getAtomType(atom_j);

                        // const size_t externalGlobalVdwType_i = molecule_i.getExternalGlobalVDWType(atom_i);
                        // const size_t externalGlobalVdwType_j = molecule_j.getExternalGlobalVDWType(atom_j);

                        // const size_t globalVdwType_i =
                        // simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_i); const size_t globalVdwType_j
                        // = simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_j);

                        const size_t globalVdwType_i = 0;
                        const size_t globalVdwType_j = 0;

                        const auto combinedIndices = {
                            moltype_i, moltype_j, atomType_i, atomType_j, globalVdwType_i, globalVdwType_j};

                        const auto coulombPreFactor = 1.0;   // TODO: implement for forcefield

                        auto [energy, force] = _coulombPotential->calculate(combinedIndices, distance, coulombPreFactor);

                        totalCoulombEnergy += energy;

                        const auto nonCoulombicPair = _nonCoulombPotential->getNonCoulombPair(combinedIndices);

                        if (const auto rncCutOff = nonCoulombicPair->getRadialCutOff(); distance < rncCutOff)
                        {
                            const auto &[nonCoulombEnergy, nonCoulombForce] = nonCoulombicPair->calculateEnergyAndForce(distance);

                            totalNonCoulombEnergy += nonCoulombEnergy;

                            force += nonCoulombForce;
                        }

                        force /= distance;

                        const auto forcexyz = force * dxyz;

                        const auto shiftForcexyz = forcexyz * txyz;

                        molecule_i.addAtomForce(atom_i, forcexyz);
                        molecule_j.addAtomForce(atom_j, -forcexyz);

                        molecule_i.addAtomShiftForce(atom_i, shiftForcexyz);
                    }
                }
            }
        }
    }

    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);
}

// // TODO: check if cutoff is smaller than smallest cell size
// /**
//  * @brief calculates forces, coulombic and non-coulombic energy for cell list routine
//  *
//  * @param simBox
//  * @param physicalData
//  * @param cellList
//  */
// inline void PotentialCellList::calculateForces(SimulationBox &simBox, PhysicalData &physicalData, CellList &cellList)
// {
//     const auto box = simBox.getBoxDimensions();

//     double totalCoulombEnergy    = 0.0;
//     double totalNonCoulombEnergy = 0.0;
//     double energy                = 0.0;

//     const double RcCutOff        = simBox.getCoulombRadiusCutOff();
//     const double RcCutOffSquared = RcCutOff * RcCutOff;

//     auto dxyz = Vec3D(0.0, 0.0, 0.0);

//     for (const auto &cell_i : cellList.getCells())
//     {
//         const auto numberOfMolecules = cell_i.getNumberOfMolecules();

//         for (size_t mol_i = 0; mol_i < numberOfMolecules; ++mol_i)
//         {
//             auto        *molecule_i = cell_i.getMolecule(mol_i);
//             const size_t moltype_i  = molecule_i->getMoltype();

//             for (size_t mol_j = 0; mol_j < mol_i; ++mol_j)
//             {
//                 auto        *molecule_j = cell_i.getMolecule(mol_j);
//                 const size_t moltype_j  = molecule_j->getMoltype();

//                 for (const size_t atom_i : cell_i.getAtomIndices(mol_i))
//                 {
//                     const size_t atomType_i = molecule_i->getAtomType(atom_i);
//                     const auto   xyz_i      = molecule_i->getAtomPosition(atom_i);

//                     for (const size_t atom_j : cell_i.getAtomIndices(mol_j))
//                     {
//                         const auto xyz_j = molecule_j->getAtomPosition(atom_j);
//                         // auto       dxyz  = xyz_i - xyz_j;

//                         dxyz[0]         = xyz_i[0] - xyz_j[0];
//                         dxyz[1]         = xyz_i[1] - xyz_j[1];
//                         dxyz[2]         = xyz_i[2] - xyz_j[2];
//                         const auto txyz = -box * round(dxyz / box);

//                         // dxyz += txyz;

//                         dxyz[0] += txyz[0];
//                         dxyz[1] += txyz[1];
//                         dxyz[2] += txyz[2];

//                         const double distanceSquared = normSquared(dxyz);

//                         if (distanceSquared > RcCutOffSquared)
//                             continue;

//                         const double distance   = ::sqrt(distanceSquared);
//                         const size_t atomType_j = molecule_j->getAtomType(atom_j);

//                         const double coulombCoefficient =
//                             simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

//                         auto force = 0.0;

//                         _coulombPotential->calcCoulomb(
//                             coulombCoefficient,
//                             distance,
//                             energy,
//                             force,
//                             simBox.getCoulombEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j),
//                             simBox.getCoulombForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

//                         totalCoulombEnergy += energy;

//                         if (const auto rncCutOff = simBox.getNonCoulombRadiusCutOff(moltype_i, moltype_j, atomType_i,
//                         atomType_j);
//                             distance < rncCutOff)
//                         {
//                             _nonCoulombPotential->calcNonCoulomb(
//                                 simBox.getGuffCoefficients(moltype_i, moltype_j, atomType_i, atomType_j),
//                                 rncCutOff,
//                                 distance,
//                                 energy,
//                                 force,
//                                 simBox.getNonCoulombEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j),
//                                 simBox.getNonCoulombForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

//                             totalNonCoulombEnergy += energy;
//                         }

//                         force /= distance;

//                         const auto forcexyz      = force * dxyz;
//                         const auto shiftForcexyz = forcexyz * txyz;

//                         molecule_i->addAtomForce(atom_i, forcexyz);
//                         molecule_j->addAtomForce(atom_j, -forcexyz);

//                         molecule_i->addAtomShiftForce(atom_i, shiftForcexyz);
//                     }
//                 }
//             }
//         }
//     }
//     for (const auto &cell_i : cellList.getCells())
//     {
//         const auto numberOfMoleculesInCell_i = cell_i.getNumberOfMolecules();

//         for (const auto *cell_j : cell_i.getNeighbourCells())
//         {
//             const auto numberOfMoleculesInCell_j = cell_j->getNumberOfMolecules();

//             for (size_t mol_i = 0; mol_i < numberOfMoleculesInCell_i; ++mol_i)
//             {
//                 auto        *molecule_i = cell_i.getMolecule(mol_i);
//                 const size_t moltype_i  = molecule_i->getMoltype();

//                 for (const auto atom_i : cell_i.getAtomIndices(mol_i))
//                 {
//                     const size_t atomType_i = molecule_i->getAtomType(atom_i);
//                     const auto   xyz_i      = molecule_i->getAtomPosition(atom_i);

//                     for (size_t mol_j = 0; mol_j < numberOfMoleculesInCell_j; ++mol_j)
//                     {
//                         auto *molecule_j = cell_j->getMolecule(mol_j);

//                         if (molecule_i == molecule_j)
//                             continue;

//                         const size_t moltype_j = molecule_j->getMoltype();

//                         for (const auto atom_j : cell_j->getAtomIndices(mol_j))
//                         {
//                             const auto xyz_j = molecule_j->getAtomPosition(atom_j);
//                             // auto       dxyz  = xyz_i - xyz_j;

//                             dxyz[0] = xyz_i[0] - xyz_j[0];
//                             dxyz[1] = xyz_i[1] - xyz_j[1];
//                             dxyz[2] = xyz_i[2] - xyz_j[2];

//                             const auto txyz = -box * round(dxyz / box);

//                             // dxyz += txyz;

//                             dxyz[0] += txyz[0];
//                             dxyz[1] += txyz[1];
//                             dxyz[2] += txyz[2];

//                             const double distanceSquared = normSquared(dxyz);

//                             if (distanceSquared > RcCutOffSquared)
//                                 continue;

//                             const double distance   = ::sqrt(distanceSquared);
//                             const size_t atomType_j = molecule_j->getAtomType(atom_j);

//                             const double coulombCoefficient =
//                                 simBox.getCoulombCoefficient(moltype_i, moltype_j, atomType_i, atomType_j);

//                             auto force = 0.0;

//                             _coulombPotential->calcCoulomb(
//                                 coulombCoefficient,
//                                 distance,
//                                 energy,
//                                 force,
//                                 simBox.getCoulombEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j),
//                                 simBox.getCoulombForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

//                             totalCoulombEnergy += energy;

//                             if (const auto rncCutOff =
//                                     simBox.getNonCoulombRadiusCutOff(moltype_i, moltype_j, atomType_i, atomType_j);
//                                 distance < rncCutOff)
//                             {
//                                 _nonCoulombPotential->calcNonCoulomb(
//                                     simBox.getGuffCoefficients(moltype_i, moltype_j, atomType_i, atomType_j),
//                                     rncCutOff,
//                                     distance,
//                                     energy,
//                                     force,
//                                     simBox.getNonCoulombEnergyCutOff(moltype_i, moltype_j, atomType_i, atomType_j),
//                                     simBox.getNonCoulombForceCutOff(moltype_i, moltype_j, atomType_i, atomType_j));

//                                 totalNonCoulombEnergy += energy;
//                             }

//                             force /= distance;

//                             const auto forcexyz      = force * dxyz;
//                             const auto shiftForcexyz = forcexyz * txyz;

//                             molecule_i->addAtomForce(atom_i, forcexyz);
//                             molecule_j->addAtomForce(atom_j, -forcexyz);

//                             molecule_i->addAtomShiftForce(atom_i, shiftForcexyz);
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     physicalData.setCoulombEnergy(totalCoulombEnergy);
//     physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);
// }