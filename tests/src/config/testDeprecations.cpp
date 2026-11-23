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

#include <gtest/gtest.h>

#include "deprecations.hpp"

namespace
{
    constexpr auto v0_9_0 = SemVer{{.major = 0, .minor = 9, .patch = 0}};
    constexpr auto v1_5_0 = SemVer{{.major = 1, .minor = 5, .patch = 0}};
    constexpr auto v3_0_0 = SemVer{{.major = 3, .minor = 0, .patch = 0}};

    constexpr Lifecycle lifecycle = {
        .deprecatedSince = v1_1_0,
        .removedIn       = v2_0_0,
    };
}   // namespace

/**
 * @brief the message defaults to an empty string
 */
TEST(TestDeprecations, defaultMessageIsEmpty)
{
    EXPECT_TRUE(lifecycle.message.empty());

    const Lifecycle withMessage = {
        .deprecatedSince = v1_1_0,
        .removedIn       = v2_0_0,
        .message         = "some message",
    };
    EXPECT_EQ(withMessage.message, "some message");
}

/**
 * @brief versions before deprecation are active
 */
TEST(TestDeprecations, statusActive)
{
    EXPECT_EQ(lifecycle.status(v0_9_0), LifecycleStatus::Active);
}

/**
 * @brief the deprecation version itself and everything up to the removal
 * version is deprecated
 */
TEST(TestDeprecations, statusDeprecated)
{
    EXPECT_EQ(lifecycle.status(v1_1_0), LifecycleStatus::Deprecated);
    EXPECT_EQ(lifecycle.status(v1_5_0), LifecycleStatus::Deprecated);
}

/**
 * @brief the removal version itself and everything after is removed
 */
TEST(TestDeprecations, statusRemoved)
{
    EXPECT_EQ(lifecycle.status(v2_0_0), LifecycleStatus::Removed);
    EXPECT_EQ(lifecycle.status(v3_0_0), LifecycleStatus::Removed);
}

/**
 * @brief status() without argument uses the current version
 */
TEST(TestDeprecations, statusCurrentVersion)
{
    EXPECT_EQ(lifecycle.status(), lifecycle.status(CURRENT_VERSION));

    const Lifecycle future = {
        .deprecatedSince = v3_0_0,
        .removedIn       = SemVer{{.major = 4, .minor = 0, .patch = 0}},
    };
    EXPECT_EQ(future.status(), LifecycleStatus::Active);
}

/**
 * @brief status is usable at compile time
 */
TEST(TestDeprecations, statusConstexpr)
{
    static_assert(lifecycle.status(v0_9_0) == LifecycleStatus::Active);
    static_assert(lifecycle.status(v1_1_0) == LifecycleStatus::Deprecated);
    static_assert(lifecycle.status(v2_0_0) == LifecycleStatus::Removed);
    SUCCEED();
}

/**
 * @brief checkWellFormed accepts removedIn >= deprecatedSince
 */
TEST(TestDeprecations, checkWellFormedValid)
{
    constexpr Lifecycle same = {.deprecatedSince = v1_1_0, .removedIn = v1_1_0};

    checkWellFormed(lifecycle);
    checkWellFormed(same);
    SUCCEED();
}

/**
 * @brief deprecate returns the lifecycle unchanged for a removal version in
 * the future
 */
TEST(TestDeprecations, deprecateReturnsLifecycle)
{
    constexpr auto result = deprecate(
        {.deprecatedSince = v1_1_0,
         .removedIn       = SemVer{{.major = 999, .minor = 0, .patch = 0}},
         .message         = "deprecated"}
    );

    EXPECT_EQ(result.deprecatedSince, v1_1_0);
    EXPECT_EQ(
        result.removedIn,
        (SemVer{{.major = 999, .minor = 0, .patch = 0}})
    );
    EXPECT_EQ(result.message, "deprecated");
    EXPECT_EQ(result.status(v1_5_0), LifecycleStatus::Deprecated);
}
