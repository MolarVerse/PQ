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

#include <algorithm>   // for for_each

#include "simulationBox.hpp"

using simulationBox::SimulationBox;

// TODO: fix issue when compiling with cuda but not with mpi
#if defined(WITH_MPI) || defined(WITH_CUDA)

/**
 * @brief flattens atom types of each atom into a single vector of size_t
 *
 * @return std::vector<size_t>
 */
std::vector<size_t> SimulationBox::flattenAtomTypes()
{
    std::vector<size_t> atomTypes;

    auto addAtomTypes = [&atomTypes](auto &atom)
    { atomTypes.push_back(atom->getAtomType()); };

    std::ranges::for_each(_atoms, addAtomTypes);

    return atomTypes;
}

/**
 * @brief flattens mol types of each molecule of each atom into a single vector
 * of size_t
 *
 * @return std::vector<size_t>
 */
std::vector<size_t> SimulationBox::flattenMolTypes()
{
    std::vector<size_t> molTypes;

    auto addMolTypes = [&molTypes](auto &molecule)
    {
        for (size_t i = 0; i < molecule.getNumberOfAtoms(); ++i)
            molTypes.push_back(molecule.getMoltype());
    };

    std::ranges::for_each(_molecules, addMolTypes);

    return molTypes;
}

/**
 * @brief flattens internal global VDW types of each atom into a single vector
 * of size_t
 *
 * @return std::vector<size_t>
 */
std::vector<size_t> SimulationBox::flattenInternalGlobalVDWTypes()
{
    std::vector<size_t> internalGlobalVDWTypes;

    auto addInternalGlobalVDWTypes = [&internalGlobalVDWTypes](auto &atom)
    { internalGlobalVDWTypes.push_back(atom->getInternalGlobalVDWType()); };

    std::ranges::for_each(_atoms, addInternalGlobalVDWTypes);

    return internalGlobalVDWTypes;
}

/**
 * @brief flattens molecule indices of each atom into a single vector of size_t
 *
 * @return std::vector<size_t>
 */
std::vector<size_t> SimulationBox::getMoleculeIndices()
{
    std::vector<size_t> moleculeIndices;

    auto addMoleculeIndices = [&moleculeIndices](auto &atom)
    { moleculeIndices.push_back(atom->getMoleculeIndex()); };

    return moleculeIndices;
}

/**
 * @brief flattens positions of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::flattenPositions()
{
    std::vector<double> positions;

    auto addPositions = [&positions](auto &atom)
    {
        const auto position = atom->getPosition();

        positions.push_back(position[0]);
        positions.push_back(position[1]);
        positions.push_back(position[2]);
    };

    std::ranges::for_each(_atoms, addPositions);

    return positions;
}

/**
 * @brief flattens velocities of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::flattenVelocities()
{
    std::vector<double> velocities;

    auto addVelocities = [&velocities](auto &atom)
    {
        const auto velocity = atom->getVelocity();

        velocities.push_back(velocity[0]);
        velocities.push_back(velocity[1]);
        velocities.push_back(velocity[2]);
    };

    std::ranges::for_each(_atoms, addVelocities);

    return velocities;
}

/**
 * @brief flattens forces of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::flattenForces()
{
    std::vector<double> forces;

    auto addForces = [&forces](auto &atom)
    {
        const auto force = atom->getForce();

        forces.push_back(force[0]);
        forces.push_back(force[1]);
        forces.push_back(force[2]);
    };

    std::ranges::for_each(_atoms, addForces);

    return forces;
}

/**
 * @brief flattens partial charges of each atom into a single vector of doubles
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::flattenPartialCharges()
{
    std::vector<double> partialCharges;

    auto addPartialCharges = [&partialCharges](auto &atom)
    { partialCharges.push_back(atom->getPartialCharge()); };

    std::ranges::for_each(_atoms, addPartialCharges);

    return partialCharges;
}

/**
 * @brief de-flattens positions of each atom from a single vector of doubles
 *
 * @param positions
 */
void SimulationBox::deFlattenPositions(const std::vector<double> &positions)
{
    size_t index = 0;

    auto setPositions = [&positions, &index](auto &atom)
    {
        linearAlgebra::Vec3D position;

        position[0] = positions[index++];
        position[1] = positions[index++];
        position[2] = positions[index++];

        atom->setPosition(position);
    };

    std::ranges::for_each(_atoms, setPositions);
}

/**
 * @brief de-flattens velocities of each atom from a single vector of doubles
 *
 * @param velocities
 */
void SimulationBox::deFlattenVelocities(const std::vector<double> &velocities)
{
    size_t index = 0;

    auto setVelocities = [&velocities, &index](auto &atom)
    {
        linearAlgebra::Vec3D velocity;

        velocity[0] = velocities[index++];
        velocity[1] = velocities[index++];
        velocity[2] = velocities[index++];

        atom->setVelocity(velocity);
    };

    std::ranges::for_each(_atoms, setVelocities);
}

/**
 * @brief de-flattens forces of each atom from a single vector of doubles
 *
 * @param forces
 */
void SimulationBox::deFlattenForces(const std::vector<double> &forces)
{
    size_t index = 0;

    auto setForces = [&forces, &index](auto &atom)
    {
        linearAlgebra::Vec3D force;

        force[0] = forces[index++];
        force[1] = forces[index++];
        force[2] = forces[index++];

        atom->setForce(force);
    };

    std::ranges::for_each(_atoms, setForces);
}

#endif   // WITH_MPI