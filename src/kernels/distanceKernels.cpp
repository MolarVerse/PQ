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
#include "unit_types.hpp"

/**
 * @brief calculate the squared distance between two particles
 *
 * @param pos_i
 * @param pos_j
 * @param simulationBox
 *
 * @return DistanceSquared The squared distance between the two particles.
 */
DistanceSquared kernel::distSquared(
    const Pos3D                 &pos_i,
    const Pos3D                 &pos_j,
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
 * @return Pos3D The distance vector between the two particles.
 */
Pos3D kernel::distVec(const Pos3D &pos_i, const Pos3D &pos_j)
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
 * @return Pos3D The distance vector between the two particles.
 */
Pos3D kernel::distVec(
    const Pos3D                 &pos_i,
    const Pos3D                 &pos_j,
    const molsys::SimulationBox &simulationBox
)
{
    auto r_ij = (pos_i - pos_j);

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
 * @return std::pair<Pos3D, DistanceSquared> The distance vector and the
 * squared distance between the two particles.
 */
std::pair<Pos3D, DistanceSquared> kernel::distVecAndDist2(
    const Pos3D &pos_i,
    const Pos3D &pos_j
)
{
    const auto r_ij = pos_i - pos_j;

    const auto rSquared = dot(r_ij, r_ij);

    return {r_ij, rSquared};
}

/**
 * @brief Calculate the distance vector and the squared distance between two
 * particles.
 *
 * @param pos_i
 * @param pos_j
 * @param simulationBox
 *
 * @return std::pair<Pos3D, DistanceSquared> The distance vector and the
 * squared distance between the two particles.
 */
std::pair<Pos3D, DistanceSquared> kernel::distVecAndDist2(
    const Pos3D                 &pos_i,
    const Pos3D                 &pos_j,
    const molsys::SimulationBox &simulationBox
)
{
    auto r_ij = pos_i - pos_j;

    simulationBox.applyPBC(r_ij);

    const auto rSquared = dot(r_ij, r_ij);

    return {r_ij, rSquared};
}
