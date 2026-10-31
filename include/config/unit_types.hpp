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

#ifndef CONFIG__UNIT_TYPES_HPP
#define CONFIG__UNIT_TYPES_HPP

/// NOTE FOR DEVELOPERS
/// This header defines type aliases for commonly used unit types in the
/// codebase. Specifically, it provides a convenient alias for 3D angular
/// quantities. Here we really only define Internal representations for 3D
/// angular quantities. External interfaces should convert to and from these
/// internal representations as needed to ensure consistency and clarity in unit
/// usage throughout the codebase.

#include <mstd/units.hpp>

#include "vector3d.hpp"

using Angles3D = linalg::Vector3D<mstd::quantity<mstd::units::rad>>;

#endif   // CONFIG__UNIT_TYPES_HPP
