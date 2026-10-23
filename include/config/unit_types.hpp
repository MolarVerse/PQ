#ifndef __UNIT_TYPES_HPP__
#define __UNIT_TYPES_HPP__

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

#endif   // __UNIT_TYPES_HPP__
