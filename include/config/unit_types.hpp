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

#ifndef __UNIT_TYPES_HPP__
#define __UNIT_TYPES_HPP__

/// NOTE FOR DEVELOPERS
/// This header defines the internal representation of 3D angular quantities:
/// a plain vector of angles in RADIANS. It deliberately does not include
/// mp-units: the unit library is expensive to parse and this header is
/// included by almost every translation unit through box.hpp. Convert from and
/// to degrees with the units from "units.hpp" in implementation files only.

#include "vector3d.hpp"

using Angles3D = linalg::Vec3D;

#endif   // __UNIT_TYPES_HPP__
