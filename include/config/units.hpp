#ifndef __UNITS_HPP__
#define __UNITS_HPP__

/// NOTE FOR DEVELOPERS
/// This header provides convenient type aliases for commonly used angular
/// units. 'rad' represents radians and 'deg' represents degrees. Use this
/// header with caution to avoid ambiguity between radians and degrees and also
/// only in cpp source files. It is recommended to include this header only in
/// implementation files (.cpp) and not in header files to prevent potential
/// conflicts and maintain clarity in unit usage throughout the codebase.

#include <mstd/units.hpp>

using rad = mstd::units::rad;
using deg = mstd::units::degree;

#endif   // __UNITS_HPP__
