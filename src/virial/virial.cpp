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
using namespace simulationBox;
using namespace physicalData;
using namespace linearAlgebra;

/**
 * @brief calculate virial for general systems
 *
 * @details It calculates the virial for all atoms in the simulation box without
 * any corrections. It already sets the virial in the physicalData object
 *
 * @param simBox
 * @param data
 */
void Virial::calculateVirial(SimulationBox &simBox, PhysicalData &data)
{
    startTimingsSection("Virial");

    _virial = {0.0};

    for (auto &molecule : simBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto forcexyz      = molecule.getAtomForce(i);
            const auto shiftForcexyz = molecule.getAtomShiftForce(i);
            const auto xyz           = molecule.getAtomPosition(i);

            const auto tensor = tensorProduct(xyz, forcexyz);

            _virial += tensor + diagonalMatrix(shiftForcexyz);

            molecule.setAtomShiftForce(i, {0.0, 0.0, 0.0});
        }
    }

    data.setVirial(_virial);

    stopTimingsSection("Virial");
}

/**
 * @brief set the virial
 *
 * @param virial
 */
void Virial::setVirial(const tensor3D &virial) { _virial = virial; }

/**
 * @brief get the virial
 *
 * @return tensor3D
 */
tensor3D Virial::getVirial() const { return _virial; }

/**
 * @brief get the virial type
 *
 * @return std::string
 */
std::string Virial::getVirialType() const { return _virialType; }