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

#include "distanceKernels.hpp"

#include "simulationBox.hpp"

/**
 * @brief calculate the squared distance between two particles
 *
 * @param pos_i
 * @param pos_j
 * @param simulationBox
 *
 * @return double The squared distance between the two particles.
 */
double kernel::distSquared(
    const linalg::Vec3D         &pos_i,
    const linalg::Vec3D         &pos_j,
    const molsys::SimulationBox &simulationBox
)
{
    auto r_ij = pos_i - pos_j;

    simulationBox.applyPBC(r_ij);

    return dot(r_ij, r_ij);
}

/**
 * @brief Calculate the distance vector between two particles.
 *
 * @param pos_i
 * @param pos_j
 *
 * @return linalg::Vec3D The distance vector between the two particles.
 */
linalg::Vec3D kernel::distVec(
    const linalg::Vec3D &pos_i,
    const linalg::Vec3D &pos_j
)
{
    return pos_i - pos_j;
}

/**
 * @brief Calculate the distance vector between two particles.
 *
 * @param pos_i
 * @param pos_j
 * @param simulationBox
 *
 * @return linalg::Vec3D The distance vector between the two particles.
 */
linalg::Vec3D kernel::distVec(
    const linalg::Vec3D         &pos_i,
    const linalg::Vec3D         &pos_j,
    const molsys::SimulationBox &simulationBox
)
{
    auto r_ij = pos_i - pos_j;

    simulationBox.applyPBC(r_ij);

    return r_ij;
}

/**
 * @brief Calculate the distance vector and the squared distance between two
 * particles.
 *
 * @param pos_i
 * @param pos_j
 *
 * @return std::pair<linalg::Vec3D, double> The distance vector and the
 * squared distance between the two particles.
 */
std::pair<linalg::Vec3D, double> kernel::distVecAndDist2(
    const linalg::Vec3D &pos_i,
    const linalg::Vec3D &pos_j
)
{
    const auto r_ij = pos_i - pos_j;

    const auto rSquared = dot(r_ij, r_ij);

    return std::make_pair(r_ij, rSquared);
}

/**
 * @brief Calculate the distance vector and the squared distance between two
 * particles.
 *
 * @param pos_i
 * @param pos_j
 * @param simulationBox
 *
 * @return std::pair<linalg::Vec3D, double> The distance vector and the
 * squared distance between the two particles.
 */
std::pair<linalg::Vec3D, double> kernel::distVecAndDist2(
    const linalg::Vec3D         &pos_i,
    const linalg::Vec3D         &pos_j,
    const molsys::SimulationBox &simulationBox
)
{
    auto r_ij = pos_i - pos_j;

    simulationBox.applyPBC(r_ij);

    const auto rSquared = dot(r_ij, r_ij);

    return std::make_pair(r_ij, rSquared);
}
