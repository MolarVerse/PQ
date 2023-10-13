#ifdef WITH_MPI

#include "simulationBox.hpp"

#include <algorithm>   // for for_each

using simulationBox::SimulationBox;

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

#endif