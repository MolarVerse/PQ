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

#include "potentialBruteForce.hpp"   // for PotentialBruteForce

#include <cstddef>   // for size_t

#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

using namespace potential;
using namespace simulationBox;
using namespace physicalData;

/**
 * @brief Destroy the Potential Brute Force:: Potential Brute Force object
 *
 */
PotentialBruteForce::~PotentialBruteForce() = default;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for brute force
 * routine
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialBruteForce::
    calculateForces(SimulationBox &simBox, PhysicalData &physicalData, CellList &)
{
    startTimingsSection("InterNonBonded");

    const auto box = simBox.getBoxPtr();

    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    // inter molecular forces
    const size_t nMol = simBox.getNumberOfMolecules();

    for (size_t mol_i = 0; mol_i < nMol; ++mol_i)
    {
        auto        &molecule_i    = simBox.getMolecule(mol_i);
        const size_t nAtomsInMol_i = molecule_i.getNumberOfAtoms();

        for (size_t mol_j = 0; mol_j < mol_i; ++mol_j)
        {
            auto        &molecule_j    = simBox.getMolecule(mol_j);
            const size_t nAtomsInMol_j = molecule_j.getNumberOfAtoms();

            for (size_t atom1 = 0; atom1 < nAtomsInMol_i; ++atom1)
            {
                for (size_t atom2 = 0; atom2 < nAtomsInMol_j; ++atom2)
                {
                    const auto [coulombEnergy, nonCoulombEnergy] =
                        calculateSingleInteraction(
                            *box,
                            molecule_i,
                            molecule_j,
                            atom1,
                            atom2
                        );

                    totalCoulombEnergy    += coulombEnergy;
                    totalNonCoulombEnergy += nonCoulombEnergy;
                }
            }
        }
    }

    physicalData.setCoulombEnergy(totalCoulombEnergy);
    physicalData.setNonCoulombEnergy(totalNonCoulombEnergy);

    simBox.flattenForces();
    simBox.flattenShiftForces();

    stopTimingsSection("InterNonBonded");
}

/**
 * @brief clone the potential
 *
 * @return std::shared_ptr<PotentialBruteForce>
 */
std::shared_ptr<Potential> PotentialBruteForce::clone() const
{
    return std::make_shared<PotentialBruteForce>(*this);
}