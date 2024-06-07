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

#include <tuple>

#include "simulationBox.hpp"
#include "vector3d.hpp"

/**
 * @brief Calculate the distance vector between two particles.
 *
 * @param pos_i
 * @param pos_j
 *
 * @return linearAlgebra::Vec3D The distance vector between the two particles.
 */
linearAlgebra::Vec3D kernel::distVec(
    const linearAlgebra::Vec3D &pos_i,
    const linearAlgebra::Vec3D &pos_j
)
{
    return pos_i - pos_j;
}

/**
 * @brief Calculate the distance vector between two particles.
 *
 * @param pos_i
 * @param pos_j
 * @param simBox
 *
 * @return linearAlgebra::Vec3D The distance vector between the two particles.
 */
linearAlgebra::Vec3D kernel::distVec(
    const linearAlgebra::Vec3D         &pos_i,
    const linearAlgebra::Vec3D         &pos_j,
    const simulationBox::SimulationBox &simBox
)
{
    auto r_ij = pos_i - pos_j;
    simBox.applyPBC(r_ij);
    return r_ij;
}

/**
 * @brief Calculate the distance vector and the squared distance between two
 * particles.
 *
 * @param pos_i
 * @param pos_j
 *
 * @return std::pair<linearAlgebra::Vec3D, double> The distance vector and the
 * squared distance between the two particles.
 */
std::pair<linearAlgebra::Vec3D, double> kernel::distVecAndDist2(
    const linearAlgebra::Vec3D &pos_i,
    const linearAlgebra::Vec3D &pos_j
)
{
    const auto r_ij = pos_i - pos_j;
    const auto r2   = dot(pos_i, pos_j);
    return std::make_pair(r_ij, r2);
}

/**
 * @brief Calculate the distance vector and the squared distance between two
 * particles.
 *
 * @param pos_i
 * @param pos_j
 * @param simBox
 *
 * @return std::pair<linearAlgebra::Vec3D, double> The distance vector and the
 * squared distance between the two particles.
 */
std::pair<linearAlgebra::Vec3D, double> kernel::distVecAndDist2(
    const linearAlgebra::Vec3D         &pos_i,
    const linearAlgebra::Vec3D         &pos_j,
    const simulationBox::SimulationBox &simBox
)
{
    auto r_ij = pos_i - pos_j;
    simBox.applyPBC(r_ij);

    const auto r2 = dot(r_ij, r_ij);

    return std::make_pair(r_ij, r2);
}
