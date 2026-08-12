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

#include "settings.hpp"
#include "simulationBox.hpp"   // for SimulationBox

using namespace virial;
using namespace simulationBox;
using namespace linearAlgebra;
using namespace pq;

using settings::Settings;
using settings::VirialType;

/**
 * @brief Calculate virial tensor and reset shift forces to zero in simulation
 * box
 *
 * @param simBox simulation box containing all atoms
 * @return tensor3D calculated virial tensor
 *
 * @details This is an overloaded version of calculateVirial that computes
 * the virial tensor for all atoms in the simulation box and returns it
 * directly without storing it in the member variable or modifying the
 * PhysicalData object. It includes contributions from both atomic forces
 * and shift forces (from periodic boundary conditions). After calculation,
 * shift forces are reset to zero. This version is useful when you need the
 * virial value without side effects on the object state.
 */
tensor3D Virial::calculateVirial(SimulationBox &simBox)
{
    tensor3D virial = {0.0};

    for (auto &atom : simBox.getAtoms())
    {
        const auto forcexyz      = atom->getForce();
        const auto shiftForcexyz = atom->getShiftForce();
        const auto xyz           = atom->getPosition();

        const auto tensor = tensorProduct(xyz, forcexyz);

        virial += tensor + diagonalMatrix(shiftForcexyz);

        atom->setShiftForce(0.0);
    }

    return virial;
}

/**
 * @brief Calculate virial contribution from QM atoms only without side effects
 *
 * @details calculates the virial tensor for QM atoms using the tensor product
 * of atomic positions and forces. This is used in hybrid QM/MM simulations to
 * compute the QM contribution to the total virial tensor.
 *
 * @warning This function assumes the center of the QM region is at the origin
 * of the box. As a result the shift forces from periodic images are taken to be
 * zero and are not considered.
 *
 * @param simBox simulation box containing QM atoms
 * @return tensor3D virial tensor from QM atoms
 */
tensor3D Virial::calculateQMVirial(const SimulationBox &simBox)
{
    tensor3D virial = {0.0};

    for (const auto &atom : simBox.getQMAtoms())
    {
        const auto forcexyz = atom->getForce();
        const auto xyz      = atom->getPosition();

        virial += tensorProduct(xyz, forcexyz);
    }

    return virial;
}

/**
 * @brief Calculate intramolecular virial correction tensor without side
 * effects
 *
 * @details Computes the intramolecular virial correction from current atomic
 * forces and positions relative to each molecule's center of mass. This
 * function only returns the correction tensor and does not modify member state
 * or PhysicalData.
 *
 * @param simBox Simulation box containing molecules
 * @return tensor3D Intramolecular virial correction tensor
 */
tensor3D Virial::intraMolecularVirialCorrection(const SimulationBox &simBox)
{
    tensor3D virial{0.0};

    if (Settings::getVirialType() == VirialType::ATOMIC)
        return virial;

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto centerOfMass = molecule.getCenterOfMass();

        for (const auto &atom : molecule.getAtoms())
        {
            const auto forcexyz = atom->getForce();
            const auto xyz      = atom->getPosition();

            auto dxyz = xyz - centerOfMass;

            simBox.applyPBC(dxyz);

            virial -= tensorProduct(dxyz, forcexyz);
        }
    }

    return virial;
}
