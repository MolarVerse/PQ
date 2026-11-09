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

#include "mp-units/systems/isq/si_quantities.h"
#include "staticMatrix.hpp"
#include "vector3d.hpp"

using Quantity   = mstd::quantity<mstd::units::one>;
using Quantity3D = linalg::Vector3D<mstd::quantity<mstd::units::one>>;
using QuantityTensor3D =
    linalg::StaticMatrix3x3<mstd::quantity<mstd::units::one>>;

using Angles3D = linalg::Vector3D<mstd::quantity<mstd::units::rad>>;
using Time     = mstd::quantity<mstd::units::fs>;

using Pressure = mstd::quantity<mstd::units::pressure[mstd::units::Pa]>;
using Pressure3D =
    linalg::Vector3D<mstd::quantity<mstd::units::pressure[mstd::units::Pa]>>;
using PressureTensor3D = linalg::StaticMatrix3x3<
    mstd::quantity<mstd::units::pressure[mstd::units::Pa]>>;
using Compressibility = mstd::quantity<
    mstd::units::compressibility[mstd::units::one / mstd::units::Pa]>;

using Temperature = mstd::units::quantity_point<
    mstd::units::thermodynamic_temperature[mstd::units::K]>;
using TemperatureDiff =
    mstd::quantity<mstd::units::thermodynamic_temperature[mstd::units::K]>;

using Frequency =
    mstd::units::quantity<mstd::units::frequency[mstd::units::Hz]>;

[[nodiscard]]
static inline Temperature toTemperature(double temperature)
{
    return mstd::units::point<mstd::units::K>(temperature);
}

[[nodiscard]]
static inline Temperature scaleAbsTemp(Temperature absTemp, double scalar)
{
    return Temperature{absTemp.quantity_from_zero() * scalar};
}

[[nodiscard]]
static inline Quantity tempRatio(Temperature absTemp, Temperature refTemp)
{
    return absTemp.quantity_from_zero() / refTemp.quantity_from_zero();
}
#endif   // CONFIG__UNIT_TYPES_HPP
