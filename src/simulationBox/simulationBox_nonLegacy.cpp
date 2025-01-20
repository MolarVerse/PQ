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

#include "simulationBox.hpp"
#include "typeAliases.hpp"
#include "vector3d.hpp"

using namespace simulationBox;
using namespace linearAlgebra;

/**
 * @brief calculate total force of simulationBox
 *
 * @return double
 */
double SimulationBox::calculateTotalForce()
{
    auto fx = 0.0;
    auto fy = 0.0;
    auto fz = 0.0;

    const auto *const forcesPtr = getForcesPtr();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for  \
                map(fx, fy, fz)                       \
                is_device_ptr(forcesPtr)              \
                reduction(+:fx, fy, fz)
#else
    #pragma omp parallel for                          \
                reduction(+:fx, fy, fz)
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms; ++i)
    {
        fx += forcesPtr[i * 3];
        fy += forcesPtr[i * 3 + 1];
        fz += forcesPtr[i * 3 + 2];
    }

    const auto totalForce = Vec3D{fx, fy, fz};

    return norm(totalForce);
}

/**
 * @brief flattens old positions of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenOldPositions()
{
    if (_oldPos.size() != _atoms.size() * 3)
        _oldPos.resize(_atoms.size() * 3);

    Real *const oldPositions = _oldPos.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            oldPositions[i * 3 + j] = atom->getPositionOld()[j];
    }
}

/**
 * @brief flattens old velocities of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenOldVelocities()
{
    if (_oldVel.size() != _atoms.size() * 3)
        _oldVel.resize(_atoms.size() * 3);

    Real *const oldVelocities = _oldVel.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            oldVelocities[i * 3 + j] = atom->getVelocityOld()[j];
    }
}

/**
 * @brief flattens old forces of each atom into a single vector of Real
 *
 */
void SimulationBox::flattenOldForces()
{
    if (_oldForces.size() != _atoms.size() * 3)
        _oldForces.resize(_atoms.size() * 3);

    Real *const oldForces = _oldForces.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            oldForces[i * 3 + j] = atom->getForceOld()[j];
    }
}

/**
 * @brief flattens atomtypes of each atom into a single vector of size_t
 * integers
 *
 * @details This method is used to flatten the atom types of each atom into a
 * single vector of size_t integers. This is useful when the atom types are
 * needed in a single vector, for example when calculating the Lennard-Jones
 * potential.
 */
void SimulationBox::flattenAtomTypes()
{
    if (_atomTypes.size() != _atoms.size())
        _atomTypes.resize(_atoms.size());

    size_t *const atomTypes = _atomTypes.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
        atomTypes[i] = _atoms[i]->getAtomType();
}

/**
 * @brief flattens molecule types of each atom into a single vector of size_t
 * integers
 *
 * @details This method is used to flatten the molecule types of each atom into
 * a single vector of size_t integers. This is useful when the molecule types
 * are needed in a single vector, for example when calculating the Lennard-Jones
 * potential.
 */
void SimulationBox::flattenMolTypes()
{
    if (_molTypes.size() != _molecules.size())
        _molTypes.resize(_molecules.size());

    size_t *const molTypes = _molTypes.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _molecules.size(); ++i)
        molTypes[i] = _molecules[i].getMoltype();
}

/**
 * @brief flattens the internal global VDW types of each atom into a single
 * vector of size_t integers
 *
 * @details This method is used to flatten the internal global VDW types of each
 * atom into a single vector of size_t integers. This is useful when the global
 * VDW types are needed in a single vector, for example when calculating the
 * Lennard-Jones potential.
 */
void SimulationBox::flattenInternalGlobalVDWTypes()
{
    if (_internalGlobalVDWTypes.size() != _atoms.size())
        _internalGlobalVDWTypes.resize(_atoms.size());

    size_t *const internalGlobalVDWTypes = _internalGlobalVDWTypes.data();

    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
        internalGlobalVDWTypes[i] = _atoms[i]->getInternalGlobalVDWType();
}

/**
 * @brief de-flattens old positions of each atom from a single vector of doubles
 * into the atom objects
 *
 */
void SimulationBox::deFlattenOldPositions()
{
    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setPositionOld(_oldPos[i * 3 + j], j);
    }
}

/**
 * @brief de-flattens old velocities of each atom from a single vector of
 * doubles into the atom objects
 *
 */
void SimulationBox::deFlattenOldVelocities()
{
    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setVelocityOld(_oldVel[i * 3 + j], j);
    }
}

/**
 * @brief de-flattens old forces of each atom from a single vector of doubles
 * into the atom objects
 *
 */
void SimulationBox::deFlattenOldForces()
{
    // clang-format off
    #pragma omp parallel for
    // clang-format on
    for (size_t i = 0; i < _atoms.size(); ++i)
    {
        const auto atom = _atoms[i];

        for (size_t j = 0; j < 3; ++j)
            atom->setForceOld(_oldForces[i * 3 + j], j);
    }
}

/**
 * @brief reset forces of all atoms
 *
 */
void SimulationBox::resetForces()
{
    flattenForces();

    auto *const _forcesPtr = getForcesPtr();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(_forcesPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms; ++i)
        for (size_t j = 0; j < 3; ++j)
            _forcesPtr[i * 3 + j] = 0.0;

    deFlattenForces();
}

/**
 * @brief get atomic scalar forces
 *
 * @return std::vector<Real>
 */
std::vector<Real> SimulationBox::getAtomicScalarForces()
{
    flattenForces();

    const auto *const forcesPtr = getForcesPtr();

    std::vector<Real> atomicScalarForces;
    atomicScalarForces.reserve(_nAtoms);

    auto *const scalarForces = atomicScalarForces.data();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(forcesPtr)             \
                map(scalarForces)
#else
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < _nAtoms; ++i)
        scalarForces[i] = ::sqrt(forcesPtr[i * 3]     * forcesPtr[i * 3]     +
                                 forcesPtr[i * 3 + 1] * forcesPtr[i * 3 + 1] +
                                 forcesPtr[i * 3 + 2] * forcesPtr[i * 3 + 2]);
    // clang-format on

    return atomicScalarForces;
}

/**
 * @brief get atomic scalar forces
 *
 * @return std::vector<Real>
 */
std::vector<Real> SimulationBox::getAtomicScalarForcesOld()
{
    flattenOldForces();

    const auto *const forcesPtr = getOldForcesPtr();

    std::vector<Real> atomicScalarForces;
    atomicScalarForces.reserve(_nAtoms);

    auto *const scalarForces = atomicScalarForces.data();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(forcesPtr)             \
                map(scalarForces)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms; ++i)
        scalarForces[i] = ::sqrt(
            forcesPtr[i * 3] * forcesPtr[i * 3] +
            forcesPtr[i * 3 + 1] * forcesPtr[i * 3 + 1] +
            forcesPtr[i * 3 + 2] * forcesPtr[i * 3 + 2]
        );

    return atomicScalarForces;
}

/**
 * @brief update old positions of all atoms
 *
 */
void SimulationBox::updateOldPositions()
{
    flattenPositions();

    const auto *const posPtr    = getPosPtr();
    auto *const       oldPosPtr = getOldPosPtr();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(posPtr, oldPosPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms; ++i)
        for (size_t j = 0; j < 3; ++j)
            oldPosPtr[i * 3 + j] = posPtr[i * 3 + j];

    deFlattenOldPositions();
}

/**
 * @brief update old velocities of all atoms
 *
 */
void SimulationBox::updateOldVelocities()
{
    flattenVelocities();

    const auto *const velPtr    = getVelPtr();
    auto *const       oldVelPtr = getOldVelPtr();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(velPtr, oldVelPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms; ++i)
        for (size_t j = 0; j < 3; ++j)
            oldVelPtr[i * 3 + j] = velPtr[i * 3 + j];

    deFlattenOldVelocities();
}

/**
 * @brief update old forces of all atoms
 *
 */
void SimulationBox::updateOldForces()
{
    flattenForces();

    const auto *const forcesPtr    = getForcesPtr();
    auto *const       oldForcesPtr = getOldForcesPtr();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                map(to:forcesPtr, oldForcesPtr)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < _nAtoms; ++i)
        for (size_t j = 0; j < 3; ++j)
            oldForcesPtr[i * 3 + j] = forcesPtr[i * 3 + j];

    deFlattenOldForces();
}