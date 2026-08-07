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
using namespace Kokkos;
using namespace linearAlgebra;

/**
 * @brief Construct a new Kokkos Simulation Box:: Kokkos Simulation Box object
 *
 * @param numAtoms
 */
KokkosSimulationBox::KokkosSimulationBox(const size_t numAtoms)
    : _atomTypes("atomTypes", numAtoms),
      _molTypes("molTypes", numAtoms),
      _moleculeIndices("moleculeIndices", numAtoms),
      _internalGlobalVDWTypes("internalGlobalVDWTypes", numAtoms),
      _positions("positions", numAtoms),
      _velocities("velocities", numAtoms),
      _forces("forces", numAtoms),
      _shiftForces("shiftForces", numAtoms),
      _partialCharges("partialCharges", numAtoms),
      _masses("masses", numAtoms),
      _boxDimensions("boxDimensions", 3)
{
}

/**
 * @brief calculate shift vector
 *
 * @param dxyz
 * @param boxDimensions
 * @param txyz
 * @return KOKKOS_FUNCTION
 */
KOKKOS_FUNCTION void KokkosSimulationBox::calcShiftVector(
    const double* dxyz,
    View<double*> boxDimensions,
    double*       txyz
)
{
    txyz[0] = -boxDimensions(0) * Kokkos::round(dxyz[0] / boxDimensions(0));
    txyz[1] = -boxDimensions(1) * Kokkos::round(dxyz[1] / boxDimensions(1));
    txyz[2] = -boxDimensions(2) * Kokkos::round(dxyz[2] / boxDimensions(2));
}

/**
 * @brief initialize simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::initKokkosSimulationBox(SimulationBox& simBox)
{
    transferAtomTypesFromSimulationBox(simBox);
    transferMolTypesFromSimulationBox(simBox);
    transferMoleculeIndicesFromSimulationBox(simBox);
    transferInternalGlobalVDWTypesFromSimulationBox(simBox);
    transferPositionsFromSimulationBox(simBox);
    transferVelocitiesFromSimulationBox(simBox);
    transferForcesFromSimulationBox(simBox);
    transferMassesFromSimulationBox(simBox);
    transferPartialChargesFromSimulationBox(simBox);
    transferBoxDimensionsFromSimulationBox(simBox);
}

/**
 * @brief initialize forces
 */
void KokkosSimulationBox::initForces()
{
    Kokkos::deep_copy(_forces.d_view, 0.0);
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
    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
        _atomTypes.h_view(i) = simBox.getAtom(i).getAtomType();

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
    auto       atom_counter = 0;
    const auto nMolecules   = simBox.getNumberOfMolecules();

    for (size_t i = 0; i < nMolecules; ++i)
    {
        const auto& molecule = simBox.getMolecule(i);

        for (size_t j = 0; j < molecule.getNumberOfAtoms(); ++j)
        {
            _molTypes.h_view(atom_counter) = molecule.getMoltype();
            ++atom_counter;
        }
    }

    Kokkos::deep_copy(_molTypes.d_view, _molTypes.h_view);
}

/**
 * @brief transfer molecule indices from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferMoleculeIndicesFromSimulationBox(
    SimulationBox& simBox
)
{
    auto       atom_counter = 0;
    const auto nMolecules   = simBox.getNumberOfMolecules();

    for (size_t i = 0; i < nMolecules; ++i)
    {
        const auto& molecule = simBox.getMolecule(i);

        for (size_t j = 0; j < molecule.getNumberOfAtoms(); ++j)
        {
            _moleculeIndices.h_view(atom_counter) = i;
            ++atom_counter;
        }
    }

    Kokkos::deep_copy(_moleculeIndices.d_view, _moleculeIndices.h_view);
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
    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto& atom                  = simBox.getAtom(i);
        _internalGlobalVDWTypes.h_view(i) = atom.getInternalGlobalVDWType();
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
    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
    {
        _positions.h_view(i, 0) = simBox.getAtom(i).getPosition()[0];
        _positions.h_view(i, 1) = simBox.getAtom(i).getPosition()[1];
        _positions.h_view(i, 2) = simBox.getAtom(i).getPosition()[2];
    }

    Kokkos::deep_copy(_positions.d_view, _positions.h_view);
}

/**
 * @brief transfer velocities from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferVelocitiesFromSimulationBox(
    SimulationBox& simBox
)
{
    const auto numberOfAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < numberOfAtoms; ++i)
    {
        _velocities.h_view(i, 0) = simBox.getAtom(i).getVelocity()[0];
        _velocities.h_view(i, 1) = simBox.getAtom(i).getVelocity()[1];
        _velocities.h_view(i, 2) = simBox.getAtom(i).getVelocity()[2];
    }

    Kokkos::deep_copy(_velocities.d_view, _velocities.h_view);
}

/**
 * @brief transfer forces from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferForcesFromSimulationBox(SimulationBox& simBox)
{
    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
    {
        _forces.h_view(i, 0) = simBox.getAtom(i).getForce()[0];
        _forces.h_view(i, 1) = simBox.getAtom(i).getForce()[1];
        _forces.h_view(i, 2) = simBox.getAtom(i).getForce()[2];
    }

    Kokkos::deep_copy(_forces.d_view, _forces.h_view);
}

/**
 * @brief transfer masses from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferMassesFromSimulationBox(SimulationBox& simBox)
{
    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
        _masses.h_view(i) = simBox.getAtom(i).getMass();

    Kokkos::deep_copy(_masses.d_view, _masses.h_view);
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
    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
        _partialCharges.h_view(i) = simBox.getAtom(i).getPartialCharge();

    Kokkos::deep_copy(_partialCharges.d_view, _partialCharges.h_view);
}

/**
 * @brief transfer box dimensions from simulation box
 *
 * @param simBox simulation box
 */
void KokkosSimulationBox::transferBoxDimensionsFromSimulationBox(
    const SimulationBox& simBox
)
{
    _boxDimensions.h_view(0) = simBox.getBoxDimensions()[0];
    _boxDimensions.h_view(1) = simBox.getBoxDimensions()[1];
    _boxDimensions.h_view(2) = simBox.getBoxDimensions()[2];

    Kokkos::deep_copy(_boxDimensions.d_view, _boxDimensions.h_view);
}

/**
 * @brief transfer positions to simulation box
 */
void KokkosSimulationBox::transferPositionsToSimulationBox(SimulationBox& simBox
)
{
    // copy positions back to host
    Kokkos::deep_copy(_positions.h_view, _positions.d_view);

    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto position = Vec3D{
            _positions.h_view(i, 0),
            _positions.h_view(i, 1),
            _positions.h_view(i, 2)
        };

        simBox.getAtom(i).setPosition(position);
    }
}

/**
 * @brief transfer velocities to simulation box
 */
void KokkosSimulationBox::transferVelocitiesToSimulationBox(
    SimulationBox& simBox
)
{
    // copy velocities back to host
    Kokkos::deep_copy(_velocities.h_view, _velocities.d_view);

    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto velocity = Vec3D{
            _velocities.h_view(i, 0),
            _velocities.h_view(i, 1),
            _velocities.h_view(i, 2)
        };

        simBox.getAtom(i).setVelocity(velocity);
    }
}

/**
 * @brief transfer forces to simulation box
 */
void KokkosSimulationBox::transferForcesToSimulationBox(SimulationBox& simBox)
{
    // copy forces back to host
    Kokkos::deep_copy(_forces.h_view, _forces.d_view);

    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto force = Vec3D{
            _forces.h_view(i, 0),
            _forces.h_view(i, 1),
            _forces.h_view(i, 2)
        };

        simBox.getAtom(i).addForce(force);
    }
}

/**
 * @brief transfer shift forces to simulation box
 */
void KokkosSimulationBox::transferShiftForcesToSimulationBox(
    SimulationBox& simBox
)
{
    // copy forces back to host
    Kokkos::deep_copy(_shiftForces.h_view, _shiftForces.d_view);

    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
    {
        const auto shiftForce = Vec3D{
            _shiftForces.h_view(i, 0),
            _shiftForces.h_view(i, 1),
            _shiftForces.h_view(i, 2)
        };

        simBox.getAtom(i).addShiftForce(shiftForce);
    }
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get atom types
 *
 * @return Kokkos::DualView<size_t*>&
 */
DualView<size_t*>& KokkosSimulationBox::getAtomTypes() { return _atomTypes; }

/**
 * @brief get molecule types
 *
 * @return Kokkos::DualView<size_t*>&
 */
DualView<size_t*>& KokkosSimulationBox::getMolTypes() { return _molTypes; }

/**
 * @brief get molecule indices
 *
 * @return Kokkos::DualView<size_t*>&
 */
DualView<size_t*>& KokkosSimulationBox::getMoleculeIndices()
{
    return _moleculeIndices;
}

/**
 * @brief get internal global VDW types
 *
 * @return Kokkos::DualView<size_t*>&
 */
DualView<size_t*>& KokkosSimulationBox::getInternalGlobalVDWTypes()
{
    return _internalGlobalVDWTypes;
}

/**
 * @brief get positions
 *
 * @return Kokkos::DualView<double* [3], Kokkos::LayoutLeft>&
 */
DualView<double* [3], Kokkos::LayoutLeft>& KokkosSimulationBox::getPositions()
{
    return _positions;
}

/**
 * @brief get velocities
 *
 * @return Kokkos::DualView<double* [3], Kokkos::LayoutLeft>&
 */
DualView<double* [3], Kokkos::LayoutLeft>& KokkosSimulationBox::getVelocities()
{
    return _velocities;
}

/**
 * @brief get forces
 *
 * @return Kokkos::DualView<double* [3], Kokkos::LayoutLeft>&
 */
DualView<double* [3], Kokkos::LayoutLeft>& KokkosSimulationBox::getForces()
{
    return _forces;
}

/**
 * @brief get shift forces
 *
 * @return Kokkos::DualView<double* [3], Kokkos::LayoutLeft>&
 */
DualView<double* [3], Kokkos::LayoutLeft>& KokkosSimulationBox::getShiftForces()
{
    return _shiftForces;
}

/**
 * @brief get masses
 *
 * @return Kokkos::DualView<double*>&
 */
DualView<double*>& KokkosSimulationBox::getMasses() { return _masses; }

/**
 * @brief get partial charges
 *
 * @return Kokkos::DualView<double*>&
 */
DualView<double*>& KokkosSimulationBox::getPartialCharges()
{
    return _partialCharges;
}

/**
 * @brief get box dimensions
 *
 * @return Kokkos::DualView<double*>
 */
DualView<double*> KokkosSimulationBox::getBoxDimensions()
{
    return _boxDimensions;
}