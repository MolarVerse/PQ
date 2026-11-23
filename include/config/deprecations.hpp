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

#ifndef CONFIG__DEPRECATIONS_HPP
#define CONFIG__DEPRECATIONS_HPP

#include <cstdint>
#include <mstd/enum.hpp>
#include <mstd/types.hpp>
#include <mstd/types/semver/semver.hpp>

#include "version.hpp"

#define LIFECYCLE_STATUS_LIST(X) \
    X(Active)                    \
    X(Deprecated)                \
    X(Removed)

MSTD_ENUM(LifecycleStatus, std::uint8_t, LIFECYCLE_STATUS_LIST)

/**
 * @brief Represents the lifecycle of a configuration key, including its
 * deprecation and removal information.
 *
 * This struct provides methods to determine the current status of the key
 * based on the provided version or the current version of the software.
 */
struct Lifecycle
{
    SemVer deprecatedSince;
    SemVer removedIn;
    // for easier struct initialization
    // NOLINTNEXTLINE(readability-redundant-string-init)
    std::string_view message = "";

    [[nodiscard]]
    constexpr LifecycleStatus status(SemVer version) const;

    [[nodiscard]]
    constexpr LifecycleStatus status() const;
};

/**
 * @brief Get the status of the key lifecycle for a specific version
 *
 * @param version The version to check against
 * @return LifecycleStatus The status of the key lifecycle for the given version
 */
constexpr LifecycleStatus Lifecycle::status(SemVer version) const
{
    if (version >= removedIn)
        return LifecycleStatus::Removed;
    if (version >= deprecatedSince)
        return LifecycleStatus::Deprecated;
    return LifecycleStatus::Active;
}

/**
 * @brief Get the status of the key lifecycle for the current version
 *
 * @return LifecycleStatus The status of the key lifecycle for the current
 * version
 */
constexpr LifecycleStatus Lifecycle::status() const
{
    return status(CURRENT_VERSION);
}

namespace detail
{
    // declared only, never defined
    void lifecycle_removedIn_is_before_deprecatedSince();
    void key_reached_removal_version_delete_it();
}   // namespace detail

/**
 * @brief Check if the key lifecycle is well-formed
 *
 * @param lifecycle The key lifecycle to check
 */
consteval void checkWellFormed(const Lifecycle &lifecycle)
{
    if (lifecycle.removedIn < lifecycle.deprecatedSince)
        detail::lifecycle_removedIn_is_before_deprecatedSince();
}

/**
 * @brief Define the deprecation and removal for a key lifecycle
 *
 * @param lifecycle The key lifecycle to define
 * @return Lifecycle The updated key lifecycle with deprecation and
 * removal defined
 */
consteval Lifecycle deprecate(Lifecycle lifecycle)
{
    checkWellFormed(lifecycle);
    if (lifecycle.removedIn <= CURRENT_VERSION)
        detail::key_reached_removal_version_delete_it();
    return lifecycle;
}

#endif   // CONFIG__DEPRECATIONS_HPP
