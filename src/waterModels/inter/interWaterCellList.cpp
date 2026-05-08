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

#include <utility>
#include <vector>

#include "atom.hpp"               // for Atom
#include "coulombPotential.hpp"   // for CoulombPotential
#include "interWater.hpp"         // for InterWater
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "typeAliases.hpp"

using namespace pq;
using namespace waterModel;

/**
 * @brief Evaluate intermolecular water interactions via cell list.
 *
 */
void InterWaterStrategyCellList::calculate(
    const InterWaterState  &state,
    SimBox                 &simBox,
    PhysicalData           &physicalData,
    const SharedCoulombPot &coulombPotential,
    CellList               &cellList
)
{
    const auto chargeProductOO = state._chargeProductOO;
    const auto chargeProductOH = state._chargeProductOH;
    const auto chargeProductHH = state._chargeProductHH;

    const auto rCut        = CoulombPot::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    const auto waterType = simBox.getWaterType();

    const auto singleInteraction = [&](Atom        &atomA,
                                       Atom        &atomB,
                                       const double chargeProduct,
                                       const auto  &nonCoulPairPtr)
    {
        if (nonCoulPairPtr)
            calculateSingleInteraction(
                atomA,
                atomB,
                chargeProduct,
                coulombPotential,
                rCutSquared,
                simBox,
                *nonCoulPairPtr,
                totalCoulombEnergy,
                totalNonCoulombEnergy
            );
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
                    !molecule_i->isActive())
                    continue;

                for (auto &atom_i : cell_i.getAtoms(mol_i))
                    for (auto &atom_j : cell_i.getAtoms(mol_j))
                    {
                        // O-H interaction
                        if (atom_i->getName() != atom_j->getName())
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                chargeProductOH,
                                state._nonCoulombPairOH
                            );
                        // O-O interaction
                        else if (atom_i->getName() == "O")
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                chargeProductOO,
                                state._nonCoulombPairOO
                            );
                        // H-H interaction
                        else
                            singleInteraction(
                                *atom_i,
                                *atom_j,
                                chargeProductHH,
                                state._nonCoulombPairHH
                            );
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

                for (auto &atom_i : cell_i.getAtoms(mol_i))
                    for (size_t mol_j = 0; mol_j < nMolsInCell_j; ++mol_j)
                    {
                        auto *molecule_j = cell_j->getMolecule(mol_j);
                        if (molecule_j->getMoltype() != waterType ||
                            !molecule_i->isActive())
                            continue;

                        if (molecule_i == molecule_j)
                            continue;

                        for (auto &atom_j : cell_j->getAtoms(mol_j))
                        {
                            // O-H interaction
                            if (atom_i->getName() != atom_j->getName())
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    chargeProductOH,
                                    state._nonCoulombPairOH
                                );
                            // O-O interaction
                            else if (atom_i->getName() == "O")
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    chargeProductOO,
                                    state._nonCoulombPairOO
                                );
                            // H-H interaction
                            else
                                singleInteraction(
                                    *atom_i,
                                    *atom_j,
                                    chargeProductHH,
                                    state._nonCoulombPairHH
                                );
                        }
                    }
            }
        }
    }
    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}
