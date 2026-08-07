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

#ifndef _DISTANCE_KERNELS_HPP_

#define _DISTANCE_KERNELS_HPP_

#include "typeAliases.hpp"
#include "vector3d.hpp"

namespace kernel
{
    [[nodiscard]] double distSquared(const pq::Vec3D &, const pq::Vec3D &, const pq::SimBox &);

    [[nodiscard]] pq::Vec3D distVec(const pq::Vec3D &, const pq::Vec3D &);

    [[nodiscard]] pq::Vec3D distVec(const pq::Vec3D &, const pq::Vec3D &, const pq::SimBox &);

    [[nodiscard]] std::pair<pq::Vec3D, double>
    distVecAndDist2(const pq::Vec3D &, const pq::Vec3D &);

    [[nodiscard]] std::pair<pq::Vec3D, double>
    distVecAndDist2(const pq::Vec3D &, const pq::Vec3D &, const pq::SimBox &);

}   // namespace kernel

#endif   // _DISTANCE_KERNELS_HPP_
