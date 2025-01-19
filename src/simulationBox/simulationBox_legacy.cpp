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

#include <algorithm>

#include "simulationBox.hpp"
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
    Vec3D totalForce(0.0);

    auto accumulateForce = [&totalForce](const auto& atom)
    { totalForce += atom->getForce(); };

    std::ranges::for_each(_atoms, accumulateForce);

    return norm(totalForce);
}

/**
 * @brief reset forces of all atoms
 *
 */
void SimulationBox::resetForces()
{
    auto resetForce = [](const auto& atom) { atom->setForceToZero(); };

    std::ranges::for_each(_atoms, resetForce);
}

/**
 * @brief get atomic scalar forces
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::getAtomicScalarForces()
{
    std::vector<double> atomicScalarForces;

    for (const auto& atom : _atoms)
        atomicScalarForces.push_back(norm(atom->getForce()));

    return atomicScalarForces;
}

/**
 * @brief get atomic scalar forces old
 *
 * @return std::vector<double>
 */
std::vector<double> SimulationBox::getAtomicScalarForcesOld()
{
    std::vector<double> atomicScalarForces;

    for (const auto& atom : _atoms)
        atomicScalarForces.push_back(norm(atom->getForceOld()));

    return atomicScalarForces;
}

/**
 * @brief update old positions of all atoms
 *
 */
void SimulationBox::updateOldPositions()
{
    auto updateOldPosition = [](const auto& atom)
    { atom->updateOldPosition(); };

    std::ranges::for_each(_atoms, updateOldPosition);
}

/**
 * @brief update old velocities of all atoms
 *
 */
void SimulationBox::updateOldVelocities()
{
    auto updateOldVelocity = [](const auto& atom)
    { atom->updateOldVelocity(); };

    std::ranges::for_each(_atoms, updateOldVelocity);
}

/**
 * @brief update old forces of all atoms
 *
 */
void SimulationBox::updateOldForces()
{
    auto updateOldForce = [](const auto& atom) { atom->updateOldForce(); };

    std::ranges::for_each(_atoms, updateOldForce);
}