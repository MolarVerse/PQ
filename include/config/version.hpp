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

#ifndef CONFIG__VERSION_HPP
#define CONFIG__VERSION_HPP

#include <mstd/types/semver/semver.hpp>

using SemVer = mstd::SemVerWithPrefix;

#define COMPILE_VERSION_ _COMPILE_VERSION_
static constexpr SemVer CURRENT_VERSION = SemVer::from_string(COMPILE_VERSION_);
#undef COMPILE_VERSION_

constexpr auto v0_0_0 = SemVer{{.major = 0, .minor = 0, .patch = 0}};
constexpr auto v1_1_0 = SemVer{{.major = 1, .minor = 1, .patch = 0}};
constexpr auto v2_0_0 = SemVer{{.major = 2, .minor = 0, .patch = 0}};

[[nodiscard]] bool isValid(const SemVer& version);

#endif   // CONFIG__VERSION_HPP
