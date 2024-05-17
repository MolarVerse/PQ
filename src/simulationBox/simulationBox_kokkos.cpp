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

#include "simulationBox_kokkos.hpp"

#include "simulationBox.hpp"   // for SimulationBox

using namespace simulationBox;

/**
 * @brief constructor
 */
KokkosSimulationBox::KokkosSimulationBox(size_t numAtoms)
    : _atomTypes("atomTypes", numAtoms),
      _molTypes("molTypes", numAtoms),
      _internalGlobalVDWTypes("internalGlobalVDWTypes", numAtoms),
      _positions("positions", numAtoms),
      _forces("forces", numAtoms),
      _partialCharges("partialCharges", numAtoms)
{
}

/**
 * @brief transfer atom types from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferAtomTypesFromSimulationBox(
    SimulationBox& simBox
)
{
    for (size_t i = 0; i < simBox.getNumberOfAtoms(); ++i)
    {
        _atomTypes.h_view(i) = simBox.getAtom(i).getAtomType();
    }

    Kokkos::deep_copy(_atomTypes.d_view, _atomTypes.h_view);
}

/**
 * @brief transfer molecule types from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferMolTypesFromSimulationBox(
    SimulationBox& simBox
)
{
    auto atom_counter = 0;

    for (size_t i = 0; i < simBox.getNumberOfMolecules(); ++i)
    {
        auto molecule = simBox.getMolecule(i);

        for (size_t j = 0; j < molecule.getNumberOfAtoms(); ++j)
        {
            _molTypes.h_view(atom_counter) = molecule.getMoltype();
            atom_counter++;
        }
    }

    Kokkos::deep_copy(_molTypes.d_view, _molTypes.h_view);
}

/**
 * @brief transfer internal global VDW types from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferInternalGlobalVDWTypesFromSimulationBox(
    SimulationBox& simBox
)
{
    for (size_t i = 0; i < simBox.getNumberOfAtoms(); ++i)
    {
        _internalGlobalVDWTypes.h_view(i) =
            simBox.getAtom(i).getInternalGlobalVDWType();
    }

    Kokkos::deep_copy(
        _internalGlobalVDWTypes.d_view,
        _internalGlobalVDWTypes.h_view
    );
}

/**
 * @brief transfer positions from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferPositionsFromSimulationBox(
    SimulationBox& simBox
)
{
    for (size_t i = 0; i < simBox.getNumberOfAtoms(); ++i)
    {
        _positions.h_view(i, 0) = simBox.getAtom(i).getPosition()[0];
        _positions.h_view(i, 1) = simBox.getAtom(i).getPosition()[1];
        _positions.h_view(i, 2) = simBox.getAtom(i).getPosition()[2];
    }

    Kokkos::deep_copy(_positions.d_view, _positions.h_view);
}

/**
 * @brief transfer partial charges from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferPartialChargesFromSimulationBox(
    SimulationBox& simBox
)
{
    for (size_t i = 0; i < simBox.getNumberOfAtoms(); ++i)
    {
        _partialCharges.h_view(i) = simBox.getAtom(i).getPartialCharge();
    }

    Kokkos::deep_copy(_partialCharges.d_view, _partialCharges.h_view);
}

/**
 * @brief initialize forces
 */
void KokkosSimulationBox::initializeForces()
{
    for (size_t i = 0; i < _forces.extent(0); ++i)
    {
        _forces.h_view(i, 0) = 0.0;
        _forces.h_view(i, 1) = 0.0;
        _forces.h_view(i, 2) = 0.0;
    }

    Kokkos::deep_copy(_forces.d_view, _forces.h_view);
}