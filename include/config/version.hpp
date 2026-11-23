#ifndef PQ__CONFIG__VERSION_HPP
#define PQ__CONFIG__VERSION_HPP

#include <mstd/types/semver/semver.hpp>

using SemVer = mstd::SemVerWithPrefix;

#define COMPILE_VERSION_ _COMPILE_VERSION_
static constexpr SemVer CURRENT_VERSION = SemVer::from_string(COMPILE_VERSION_);
#undef COMPILE_VERSION_

constexpr auto v1_1_0 = SemVer{{.major = 1, .minor = 1, .patch = 0}};
constexpr auto v2_0_0 = SemVer{{.major = 2, .minor = 0, .patch = 0}};

#endif   // PQ__CONFIG__VERSION_HPP
