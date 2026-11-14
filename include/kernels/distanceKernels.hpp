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

#ifndef KERNELS__DISTANCE_KERNELS_HPP
#define KERNELS__DISTANCE_KERNELS_HPP

#include "unit_types.hpp"
#include "vector3d.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace kernel
{
    [[nodiscard]]
    double distSquared(
        const linalg::Vec3D &,
        const linalg::Vec3D &,
        const molsys::SimulationBox &
    );

    [[nodiscard]]
    Pos3D distVec(const Pos3D &, const Pos3D &);

    [[nodiscard]]
    Pos3D distVec(const Pos3D &, const Pos3D &, const molsys::SimulationBox &);

    [[nodiscard]]
    std::pair<linalg::Vec3D, double> distVecAndDist2(
        const linalg::Vec3D &,
        const linalg::Vec3D &
    );

    [[nodiscard]] std::pair<linalg::Vec3D, double> distVecAndDist2(
        const linalg::Vec3D &,
        const linalg::Vec3D &,
        const molsys::SimulationBox &
    );

}   // namespace kernel

#endif   // KERNELS__DISTANCE_KERNELS_HPP
