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

#ifndef UTILITIES__UNIT_TYPES_HPP
#define UTILITIES__UNIT_TYPES_HPP

/// NOTE FOR DEVELOPERS
/// This header defines type aliases for commonly used unit types in the
/// codebase. Specifically, it provides a convenient alias for 3D angular
/// quantities. Here we really only define Internal representations for 3D
/// angular quantities. External interfaces should convert to and from these
/// internal representations as needed to ensure consistency and clarity in unit
/// usage throughout the codebase.

#include <algorithm>
#include <cmath>
#include <mstd/units.hpp>

#include "staticMatrix.hpp"
#include "vector3d.hpp"

using Quantity   = mstd::quantity<mstd::units::one>;
using Quantity3D = linalg::Vector3D<mstd::quantity<mstd::units::one>>;
using QuantityTensor3D =
    linalg::StaticMatrix3x3<mstd::quantity<mstd::units::one>>;

using Angle   = mstd::quantity<mstd::units::rad>;
using Angle3D = linalg::Vector3D<mstd::quantity<mstd::units::rad>>;

using Time        = mstd::quantity<mstd::units::fs>;
using TimeSquared = mstd::quantity<mstd::units::fs * mstd::units::fs>;
using TimeMs      = mstd::quantity<mstd::units::ms>;

using Pressure = mstd::quantity<mstd::units::pressure[mstd::units::Pa]>;
using Pressure3D =
    linalg::Vector3D<mstd::quantity<mstd::units::pressure[mstd::units::Pa]>>;
using PressureTensor3D = linalg::StaticMatrix3x3<
    mstd::quantity<mstd::units::pressure[mstd::units::Pa]>>;
using Compressibility = mstd::quantity<
    mstd::units::compressibility[mstd::units::one / mstd::units::Pa]>;

using Temperature =
    mstd::units::quantity_point<mstd::units::temperature[mstd::units::K]>;
using TemperatureDiff =
    mstd::quantity<mstd::units::temperature[mstd::units::K]>;

using Frequency =
    mstd::units::quantity<mstd::units::frequency[mstd::units::Hz]>;
using Distance      = mstd::quantity<mstd::units::angstrom>;
using DistanceSixth = mstd::quantity<
    mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom *
    mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom>;
using DistanceTwelfth =
    mstd::quantity<DistanceSixth::unit * DistanceSixth::unit>;

using Energy       = mstd::quantity<mstd::units::kcal>;
using EnergyPerMol = mstd::quantity<mstd::units::kcal_per_mol>;
using C6           = mstd::quantity<EnergyPerMol::unit * DistanceSixth::unit>;
using C12          = mstd::quantity<EnergyPerMol::unit * DistanceTwelfth::unit>;
using Virial       = linalg::StaticMatrix3x3<Energy>;
using VirialPerMol = linalg::StaticMatrix3x3<EnergyPerMol>;
using Action       = mstd::quantity<EnergyPerMol::unit * Time::unit>;

using Mass = mstd::quantity<mstd::units::g / mstd::units::mol>;

// TODO: make Distance and Pos incompatible
using InvDistance = mstd::quantity<mstd::units::one / mstd::units::angstrom>;
using DistanceSquared =
    mstd::quantity<mstd::units::angstrom * mstd::units::angstrom>;
using InvDistanceSquared = mstd::quantity<
    mstd::units::one / (mstd::units::angstrom * mstd::units::angstrom)>;
using InvDistanceCubed = mstd::quantity<
    mstd::units::one /
    (mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom)>;

using Pos = mstd::quantity<mstd::units::angstrom>;
using PosMass =
    mstd::quantity<mstd::units::angstrom * mstd::units::g / mstd::units::mol>;
using PosMass3D      = linalg::Vector3D<PosMass>;
using InvPos         = mstd::quantity<mstd::units::one / mstd::units::angstrom>;
using Pos3D          = linalg::Vector3D<Pos>;
using PosTensor3D    = linalg::StaticMatrix3x3<Pos>;
using InvPosTensor3D = linalg::StaticMatrix3x3<InvPos>;

using Force          = mstd::quantity<EnergyPerMol::unit / Distance::unit>;
using ForceSquared   = mstd::quantity<Force::unit * Force::unit>;
using Force3D        = linalg::Vector3D<Force>;
using Force3DSquared = linalg::Vector3D<ForceSquared>;

// synonym for Energy
using ShiftForce   = mstd::quantity<Force::unit * Distance::unit>;
using ShiftForce3D = linalg::Vector3D<ShiftForce>;

using PosSquaredInvMass = mstd::quantity<
    mstd::units::angstrom * mstd::units::angstrom /
    (mstd::units::g / mstd::units::mol)>;

using MassInvPosSquared = mstd::quantity<
    mstd::units::g / mstd::units::mol /
    (mstd::units::angstrom * mstd::units::angstrom)>;

using InertiaTensor3D = linalg::StaticMatrix3x3<mstd::quantity<
    mstd::units::angstrom * mstd::units::angstrom * mstd::units::g /
    mstd::units::mol>>;

using MassPerTime =
    mstd::quantity<mstd::units::g / mstd::units::mol / mstd::units::s>;
using MassPerTimeSquared = mstd::quantity<
    mstd::units::g / mstd::units::mol / (mstd::units::s * mstd::units::s)>;
using Volume = mstd::quantity<
    mstd::units::angstrom * mstd::units::angstrom * mstd::units::angstrom>;
using InvVolume = mstd::quantity<
    mstd::units::one / (mstd::units::volume
                            [mstd::units::angstrom * mstd::units::angstrom *
                             mstd::units::angstrom])>;

using Vel        = mstd::quantity<mstd::units::angstrom / mstd::units::s>;
using VelSquared = mstd::quantity<
    (mstd::units::angstrom / mstd::units::s) *
    (mstd::units::angstrom / mstd::units::s)>;
using Vel3D        = linalg::Vector3D<Vel>;
using VelSquared3D = linalg::Vector3D<VelSquared>;

using PosVel = mstd::quantity<
    mstd::units::angstrom * mstd::units::angstrom / mstd::units::s>;
using PosVel3D = linalg::Vector3D<PosVel>;

using Momentum =
    mstd::quantity<mstd::units::angstrom * Mass::unit / mstd::units::s>;
using Momentum3D = linalg::Vector3D<Momentum>;

using AngularMomentum = mstd::quantity<
    mstd::units::angstrom * mstd::units::angstrom * mstd::units::g /
    mstd::units::mol / mstd::units::s>;
using AngularMomentum3D = linalg::Vector3D<AngularMomentum>;

using ForceConstant = mstd::quantity<
    mstd::units::kcal_per_mol / mstd::units::angstrom / mstd::units::angstrom>;

using Hessian = std::vector<std::vector<ForceConstant>>;

using ForceConstantPerTime =
    mstd::quantity<ForceConstant::unit / mstd::units::fs>;

using AngleForceConstant = mstd::quantity<
    mstd::units::kcal_per_mol / mstd::units::rad / mstd::units::rad>;

using MassVelSquared = mstd::quantity<
    mstd::units::g / mstd::units::mol * mstd::units::angstrom *
    mstd::units::angstrom / (mstd::units::s * mstd::units::s)>;

using MassVelSquaredTensor3D = linalg::StaticMatrix3x3<MassVelSquared>;

using Charge = mstd::quantity<
    mstd::units::electric_charge[mstd::units::elementary_charge]>;
using ChargeSquared = mstd::quantity<
    (mstd::units::electric_charge[mstd::units::elementary_charge]) *
    (mstd::units::electric_charge[mstd::units::elementary_charge])>;

using EnergyLengthPerMol =
    mstd::quantity<EnergyPerMol::unit * mstd::units::angstrom>;

using Stress =
    linalg::StaticMatrix3x3<mstd::quantity<EnergyPerMol::unit / Volume::unit>>;

using Density = mstd::quantity<mstd::units::kg / mstd::units::litre>;

using Percent = mstd::quantity<mstd::units::percent>;

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
/**
 * @brief angle between two vectors of quantities (e.g. Pos3D x Pos3D)
 *
 * @details the cosine is the dimensionless ratio dot / (|lhs| * |rhs|),
 * clamped to [-1, 1] against rounding errors before the arccosine is taken
 *
 * @return angle in the range [0, pi]
 */
template <pq::QuantityLikeVector3D U>
[[nodiscard]]
static inline Angle angle(const U &lhs, const U &rhs)
{
    const auto cosine = (dot(lhs, rhs) / (norm(lhs) * norm(rhs)))
                            .numerical_value_in(mstd::units::one);

    return std::acos(std::clamp(cosine, -1.0, 1.0)) * mstd::units::rad;
}

#endif   // UTILITIES__UNIT_TYPES_HPP
