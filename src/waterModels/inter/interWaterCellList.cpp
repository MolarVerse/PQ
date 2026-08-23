/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <algorithm>
#include <vector>

#include "atom.hpp"   // for Atom
#include "atomNumberMap.hpp"
#include "celllist.hpp"     // for CellList
#include "interWater.hpp"   // for InterWater
#include "physicalData.hpp"
#include "potential.hpp"   // for ChargeTag

using namespace constants;
using namespace potential;
using namespace pq;
using namespace waterModel;
using namespace physicalData;
using namespace molsys;

namespace
{
    const auto oxygenAtomicNumber = atomNumberMap.at("o");
}   // namespace

/**
 * @brief Evaluate intermolecular water interactions via cell list.
 *
 */
void InterWaterStrategyCellList::calculate(
    const InterWaterState                              &state,
    molsys::SimulationBox                              &simBox,
    physicalData::PhysicalData                         &physicalData,
    const std::shared_ptr<potential::CoulombPotential> &coulombPotential,
    molsys::CellList                                   &cellList
)
{
    const auto rCut = potential::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto waterType = simBox.getWaterType();

    const auto singleInteraction =
        [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
    {
        if (nonCoulPairPtr)
        {
            calculateSingleInteraction<MMChargeTag, MMChargeTag>(
                atomA,
                atomB,
                coulombPotential,
                rCutSquared,
                simBox,
                *nonCoulPairPtr,
                totalCoulombEnergy,
                totalNonCoulombEnergy
            );
        }
    };

    for (const auto &cell_i : cellList.getCells())
    {
        const auto nMols = cell_i.getNumberOfMolecules();

        for (size_t mol_i = 0; mol_i < nMols; ++mol_i)
        {
            auto *molecule_i = cell_i.getMolecule(mol_i);
            if (molecule_i->getMoltype() != waterType ||
                !molecule_i->isActive())
                continue;

            for (size_t mol_j = 0; mol_j < mol_i; ++mol_j)
            {
                auto *molecule_j = cell_i.getMolecule(mol_j);
                if (molecule_j->getMoltype() != waterType ||
                    !molecule_j->isActive())
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (auto *atom_j : cell_i.getAtoms(mol_j))
                    {
                        const bool isAtom_j_O =
                            atom_j->getAtomicNumber() == oxygenAtomicNumber;

                        // O-H interaction (different atom types)
                        if (isAtom_i_O != isAtom_j_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOH
                            );
                            // O-O interaction
                        }
                        else if (isAtom_i_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOO
                            );
                            // H-H interaction
                        }
                        else
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairHH
                            );
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto nMolsInCell_i = cell_i.getNumberOfMolecules();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto nMolsInCell_j = cell_j->getNumberOfMolecules();

            for (size_t mol_i = 0; mol_i < nMolsInCell_i; ++mol_i)
            {
                auto *molecule_i = cell_i.getMolecule(mol_i);
                if (molecule_i->getMoltype() != waterType ||
                    !molecule_i->isActive())
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (size_t mol_j = 0; mol_j < nMolsInCell_j; ++mol_j)
                    {
                        auto *molecule_j = cell_j->getMolecule(mol_j);
                        if (molecule_j->getMoltype() != waterType ||
                            !molecule_j->isActive())
                            continue;

                        if (molecule_i == molecule_j)
                            continue;

                        for (auto *atom_j : cell_j->getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

/**
 * @brief Compute core-to-outer Coulomb interactions using the cell list.
 *
 * @param state Inter-water parameters.
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 * @param cellList Cell list structure used for neighbor searching.
 */
void InterWaterStrategyCellList::calculateCoreToOuterForces(
    const InterWaterState & /*state*/,
    molsys::SimulationBox                              &simBox,
    PhysicalData                                       &physicalData,
    const std::shared_ptr<potential::CoulombPotential> &coulombPotential,
    molsys::CellList                                   &cellList
)
{
    const auto rCut = potential::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy = 0.0;

    const auto singleCoulombInteraction = [&](Atom &atomA, Atom &atomB)
    {
        calculateSingleCoulombInteraction<QMChargeTag, MMChargeTag>(
            atomA,
            atomB,
            coulombPotential,
            rCutSquared,
            simBox,
            totalCoulombEnergy
        );
    };

    const auto isNonWaterMolecule =
        [](const std::vector<size_t> &waterMolecules,
           const size_t               molIndex) -> bool
    {
        return std::ranges::find(waterMolecules, molIndex) ==
               waterMolecules.end();
    };

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules = cell_i.getWaterMoleculeIndices();

        for (const auto mol_i : cell_i.getCoreMoleculeIndices())
        {
            if (isNonWaterMolecule(waterMolecules, mol_i))
                continue;

            for (const auto mol_j : cell_i.getActiveMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules, mol_j))
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                    for (auto *atom_j : cell_i.getAtoms(mol_j))
                        singleCoulombInteraction(*atom_i, *atom_j);
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_i.getCoreMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_i, mol_i))
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    for (const auto mol_j : cell_j->getActiveMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_j, mol_j))
                            continue;

                        for (auto *atom_j : cell_j->getAtoms(mol_j))
                            singleCoulombInteraction(*atom_i, *atom_j);
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_j->getCoreMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_j, mol_i))
                    continue;

                for (auto *atom_i : cell_j->getAtoms(mol_i))
                {
                    for (const auto mol_j : cell_i.getActiveMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_i, mol_j))
                            continue;

                        for (auto *atom_j : cell_i.getAtoms(mol_j))
                            singleCoulombInteraction(*atom_i, *atom_j);
                    }
                }
            }
        }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
}

/**
 * @brief Compute layer-to-outer interactions using the cell list.
 *
 * @param state Inter-water parameters.
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 * @param cellList Cell list structure used for neighbor searching.
 */
void InterWaterStrategyCellList::calculateLayerToOuterForces(
    const InterWaterState                              &state,
    molsys::SimulationBox                              &simBox,
    PhysicalData                                       &physicalData,
    const std::shared_ptr<potential::CoulombPotential> &coulombPotential,
    molsys::CellList                                   &cellList
)
{
    const auto rCut = potential::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto singleInteraction =
        [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
    {
        if (nonCoulPairPtr)
        {
            calculateSingleInteraction<QMChargeTag, MMChargeTag>(
                atomA,
                atomB,
                coulombPotential,
                rCutSquared,
                simBox,
                *nonCoulPairPtr,
                totalCoulombEnergy,
                totalNonCoulombEnergy
            );
        }
    };

    const auto isNonWaterMolecule =
        [](const std::vector<size_t> &waterMolecules,
           const size_t               molIndex) -> bool
    {
        return std::ranges::find(waterMolecules, molIndex) ==
               waterMolecules.end();
    };

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules = cell_i.getWaterMoleculeIndices();

        for (const auto mol_i : cell_i.getInactiveNonCoreMoleculeIndices())
        {
            if (isNonWaterMolecule(waterMolecules, mol_i))
                continue;

            for (const auto mol_j : cell_i.getActiveMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules, mol_j))
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (auto *atom_j : cell_i.getAtoms(mol_j))
                    {
                        const bool isAtom_j_O =
                            atom_j->getAtomicNumber() == oxygenAtomicNumber;

                        // O-H interaction (different atom types)
                        if (isAtom_i_O != isAtom_j_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOH
                            );
                            // O-O interaction
                        }
                        else if (isAtom_i_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOO
                            );
                            // H-H interaction
                        }
                        else
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairHH
                            );
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_i.getInactiveNonCoreMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_i, mol_i))
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (const auto mol_j : cell_j->getActiveMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_j, mol_j))
                            continue;

                        for (auto *atom_j : cell_j->getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_j->getInactiveNonCoreMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_j, mol_i))
                    continue;

                for (auto *atom_i : cell_j->getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (const auto mol_j : cell_i.getActiveMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_i, mol_j))
                            continue;

                        for (auto *atom_j : cell_i.getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

/**
 * @brief Compute outer-to-outer interactions using the cell list.
 *
 * @param state Inter-water parameters.
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 * @param cellList Cell list structure used for neighbor searching.
 */
void InterWaterStrategyCellList::calculateOuterToOuterForces(
    const InterWaterState                              &state,
    molsys::SimulationBox                              &simBox,
    physicalData::PhysicalData                         &physicalData,
    const std::shared_ptr<potential::CoulombPotential> &coulombPotential,
    molsys::CellList                                   &cellList
)
{
    const auto rCut = potential::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto singleInteraction =
        [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
    {
        if (nonCoulPairPtr)
        {
            calculateSingleInteraction<MMChargeTag, MMChargeTag>(
                atomA,
                atomB,
                coulombPotential,
                rCutSquared,
                simBox,
                *nonCoulPairPtr,
                totalCoulombEnergy,
                totalNonCoulombEnergy
            );
        }
    };

    const auto isNonWaterMolecule =
        [](const std::vector<size_t> &waterMolecules,
           const size_t               molIndex) -> bool
    {
        return std::ranges::find(waterMolecules, molIndex) ==
               waterMolecules.end();
    };

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules = cell_i.getWaterMoleculeIndices();

        for (const auto mol_i : cell_i.getActiveMoleculeIndices())
        {
            if (isNonWaterMolecule(waterMolecules, mol_i))
                continue;

            for (const auto mol_j : cell_i.getActiveMoleculeIndices())
            {
                if (mol_j >= mol_i)
                    break;

                if (isNonWaterMolecule(waterMolecules, mol_j))
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (auto *atom_j : cell_i.getAtoms(mol_j))
                    {
                        const bool isAtom_j_O =
                            atom_j->getAtomicNumber() == oxygenAtomicNumber;

                        // O-H interaction (different atom types)
                        if (isAtom_i_O != isAtom_j_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOH
                            );
                            // O-O interaction
                        }
                        else if (isAtom_i_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOO
                            );
                            // H-H interaction
                        }
                        else
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairHH
                            );
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_i.getActiveMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_i, mol_i))
                    continue;

                auto *molecule_i = cell_i.getMolecule(mol_i);

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (const auto mol_j : cell_j->getActiveMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_j, mol_j))
                            continue;

                        auto *molecule_j = cell_j->getMolecule(mol_j);

                        if (molecule_i == molecule_j)
                            continue;

                        for (auto *atom_j : cell_j->getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

/**
 * @brief Compute smoothing-zone interactions against MM molecules.
 *
 * @param state Inter-water parameters.
 * @param simBox Simulation box containing molecules.
 * @param physicalData Physical data to store energy results.
 * @param coulombPotential Coulomb potential evaluator.
 * @param cellList Cell list structure used for neighbor searching.
 */
void InterWaterStrategyCellList::calculateHotspotSmoothingMMForces(
    const InterWaterState                              &state,
    molsys::SimulationBox                              &simBox,
    physicalData::PhysicalData                         &physicalData,
    const std::shared_ptr<potential::CoulombPotential> &coulombPotential,
    molsys::CellList                                   &cellList
)
{
    const auto rCut = potential::CoulombPotential::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto singleInteraction =
        [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
    {
        if (nonCoulPairPtr)
        {
            calculateSingleInteraction<MMChargeTag, QMChargeTag>(
                atomA,
                atomB,
                coulombPotential,
                rCutSquared,
                simBox,
                *nonCoulPairPtr,
                totalCoulombEnergy,
                totalNonCoulombEnergy
            );
        }
    };

    const auto singleInteractionOneWay =
        [&](Atom &atomA, Atom &atomB, const auto &nonCoulPairPtr)
    {
        if (nonCoulPairPtr)
        {
            calculateSingleInteractionOneWay<MMChargeTag, QMChargeTag>(
                atomA,
                atomB,
                coulombPotential,
                rCutSquared,
                simBox,
                *nonCoulPairPtr,
                totalCoulombEnergy,
                totalNonCoulombEnergy
            );
        }
    };

    const auto isNonWaterMolecule =
        [](const std::vector<size_t> &waterMolecules,
           const size_t               molIndex) -> bool
    {
        return std::ranges::find(waterMolecules, molIndex) ==
               waterMolecules.end();
    };

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules = cell_i.getWaterMoleculeIndices();

        for (const auto mol_i : cell_i.getSmoothingMoleculeIndices())
        {
            if (isNonWaterMolecule(waterMolecules, mol_i))
                continue;

            for (const auto mol_j : cell_i.getNonSmoothingMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules, mol_j))
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (auto *atom_j : cell_i.getAtoms(mol_j))
                    {
                        const bool isAtom_j_O =
                            atom_j->getAtomicNumber() == oxygenAtomicNumber;

                        // O-H interaction (different atom types)
                        if (isAtom_i_O != isAtom_j_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOH
                            );
                            // O-O interaction
                        }
                        else if (isAtom_i_O)
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOO
                            );
                            // H-H interaction
                        }
                        else
                        {
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairHH
                            );
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_i.getSmoothingMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_i, mol_i))
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (const auto mol_j :
                         cell_j->getNonSmoothingMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_j, mol_j))
                            continue;

                        for (auto *atom_j : cell_j->getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_j->getSmoothingMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_j, mol_i))
                    continue;

                for (auto *atom_i : cell_j->getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (const auto mol_j :
                         cell_i.getNonSmoothingMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_i, mol_j))
                            continue;

                        for (auto *atom_j : cell_i.getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules = cell_i.getWaterMoleculeIndices();

        for (const auto mol_i : cell_i.getSmoothingMoleculeIndices())
        {
            if (isNonWaterMolecule(waterMolecules, mol_i))
                continue;

            for (const auto mol_j : cell_i.getSmoothingMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules, mol_j))
                    continue;

                if (mol_i == mol_j)
                    continue;

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (auto *atom_j : cell_i.getAtoms(mol_j))
                    {
                        const bool isAtom_j_O =
                            atom_j->getAtomicNumber() == oxygenAtomicNumber;

                        // O-H interaction (different atom types)
                        if (isAtom_i_O != isAtom_j_O)
                        {
                            singleInteractionOneWay(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOH
                            );
                            // O-O interaction
                        }
                        else if (isAtom_i_O)
                        {
                            singleInteractionOneWay(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairOO
                            );
                            // H-H interaction
                        }
                        else
                        {
                            singleInteractionOneWay(
                                *atom_i,
                                *atom_j,
                                state._nonCoulombPairHH
                            );
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_i.getSmoothingMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_i, mol_i))
                    continue;

                auto *molecule_i = cell_i.getMolecule(mol_i);

                for (auto *atom_i : cell_i.getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (const auto mol_j :
                         cell_j->getSmoothingMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_j, mol_j))
                            continue;

                        auto *molecule_j = cell_j->getMolecule(mol_j);

                        if (molecule_i == molecule_j)
                            continue;

                        for (auto *atom_j : cell_j->getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteractionOneWay(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteractionOneWay(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteractionOneWay(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto &cell_i : cellList.getCells())
    {
        const auto &waterMolecules_i = cell_i.getWaterMoleculeIndices();

        for (const auto *cell_j : cell_i.getNeighbourCells())
        {
            const auto &waterMolecules_j = cell_j->getWaterMoleculeIndices();

            for (const auto mol_i : cell_j->getSmoothingMoleculeIndices())
            {
                if (isNonWaterMolecule(waterMolecules_j, mol_i))
                    continue;

                auto *molecule_i = cell_j->getMolecule(mol_i);

                for (auto *atom_i : cell_j->getAtoms(mol_i))
                {
                    const bool isAtom_i_O =
                        atom_i->getAtomicNumber() == oxygenAtomicNumber;
                    for (const auto mol_j :
                         cell_i.getSmoothingMoleculeIndices())
                    {
                        if (isNonWaterMolecule(waterMolecules_i, mol_j))
                            continue;

                        auto *molecule_j = cell_i.getMolecule(mol_j);

                        if (molecule_i == molecule_j)
                            continue;

                        for (auto *atom_j : cell_i.getAtoms(mol_j))
                        {
                            const bool isAtom_j_O =
                                atom_j->getAtomicNumber() == oxygenAtomicNumber;

                            // O-H interaction (different atom types)
                            if (isAtom_i_O != isAtom_j_O)
                            {
                                singleInteractionOneWay(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOH
                                );
                                // O-O interaction
                            }
                            else if (isAtom_i_O)
                            {
                                singleInteractionOneWay(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairOO
                                );
                                // H-H interaction
                            }
                            else
                            {
                                singleInteractionOneWay(
                                    *atom_i,
                                    *atom_j,
                                    state._nonCoulombPairHH
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}
