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

#include "virial.hpp"

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData, physicalData, simulationBox
#include "simulationBox.hpp"   // for SimulationBox

using namespace virial;

/**
 * @brief clones the molecular virial object
 *
 * @return std::shared_ptr<Virial>
 */
std::shared_ptr<Virial> VirialMolecular::clone() const
{
    return std::make_shared<VirialMolecular>(*this);
}

/**
 * @brief clones the atomic virial object
 *
 */
std::shared_ptr<Virial> VirialAtomic::clone() const
{
    return std::make_shared<VirialAtomic>(*this);
}

/**
 * @brief calculate virial for general systems
 *
 * @details It calculates the virial for all atoms in the simulation box without
 * any corrections. It already sets the virial in the physicalData object
 *
 * @param simulationBox
 * @param physicalData
 */
void Virial::calculateVirial(
    simulationBox::SimulationBox &simulationBox,
    physicalData::PhysicalData   &physicalData
)
{
    startTimingsSection("Virial");

    _virial = {0.0};

    for (auto &molecule : simulationBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto forcexyz      = molecule.getAtomForce(i);
            const auto shiftForcexyz = molecule.getAtomShiftForce(i);
            const auto xyz           = molecule.getAtomPosition(i);

            // TODO: check if correct with shiftForcexyz
            _virial +=
                tensorProduct(xyz, forcexyz) + diagonalMatrix(shiftForcexyz);

            molecule.setAtomShiftForce(i, {0.0, 0.0, 0.0});
        }
    }

    physicalData.setVirial(_virial);

    stopTimingsSection("Virial");
}

/**
 * @brief calculate virial for molecular systems
 *
 * @details it calls the general virial calculation and then corrects it for
 *          intramolecular interactions. Afterwards it sets the virial in the
 *          physicalData object
 *
 * @param simulationBox
 * @param physicalData
 */
void VirialMolecular::calculateVirial(
    simulationBox::SimulationBox &simulationBox,
    physicalData::PhysicalData   &physicalData
)
{
    Virial::calculateVirial(simulationBox, physicalData);

    physicalData.setVirial(_virial);
}

/**
 * @brief calculate intramolecular virial correction
 *
 * @note it directly corrects the virial member variable
 *
 * @param simulationBox
 */
void VirialMolecular::intraMolecularVirialCorrection(
    simulationBox::SimulationBox &simulationBox,
    physicalData::PhysicalData   &physicalData
)
{
    startTimingsSection("IntraMolecular Correction");

    _virial = {0.0};

    for (const auto &molecule : simulationBox.getMolecules())
    {
        const auto   centerOfMass  = molecule.getCenterOfMass();
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto forcexyz = molecule.getAtomForce(i);
            const auto xyz      = molecule.getAtomPosition(i);

            auto dxyz = xyz - centerOfMass;

            simulationBox.applyPBC(dxyz);

            _virial -= tensorProduct(dxyz, forcexyz);
        }
    }

    physicalData.addVirial(_virial);

    stopTimingsSection("IntraMolecular Correction");
}