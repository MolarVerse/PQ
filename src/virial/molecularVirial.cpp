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

#include "molecularVirial.hpp"

#include "physicalData.hpp"
#include "simulationBox.hpp"

using namespace virial;
using namespace simulationBox;
using namespace physicalData;
using namespace linearAlgebra;

/**
 * @brief Construct a new Virial Molecular:: Virial Molecular object
 *
 */
MolecularVirial::MolecularVirial() : Virial() { _virialType = "molecular"; }

/**
 * @brief clones the molecular virial object
 *
 * @return std::shared_ptr<Virial>
 */
std::shared_ptr<Virial> MolecularVirial::clone() const
{
    return std::make_shared<MolecularVirial>(*this);
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
void MolecularVirial::calculateVirial(
    SimulationBox &simulationBox,
    PhysicalData  &physicalData
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
 * @param simBox
 * @param data
 */
void MolecularVirial::intraMolecularVirialCorrection(
    SimulationBox &simBox,
    PhysicalData  &data
)
{
    startTimingsSection("IntraMolecular Correction");

    _virial = {0.0};

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto   centerOfMass  = molecule.getCenterOfMass();
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto forcexyz = molecule.getAtomForce(i);
            const auto xyz      = molecule.getAtomPosition(i);

            auto dxyz = xyz - centerOfMass;

            simBox.applyPBC(dxyz);

            _virial -= tensorProduct(dxyz, forcexyz);
        }
    }

    data.addVirial(_virial);

    stopTimingsSection("IntraMolecular Correction");
}