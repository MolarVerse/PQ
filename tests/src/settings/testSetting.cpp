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

#include <string>
#include <type_traits>

#include "exceptions.hpp"
#include "setting.hpp"
#include "throwWithMessage.hpp"

namespace
{
    enum class Color : std::uint8_t
    {
        RED,
        GREEN,
        BLUE
    };

    // ---------------------------------------------------------------------
    // Typed tests: behavior must be identical for every value type
    // ---------------------------------------------------------------------

    template <typename T>
    struct SampleValues;

    template <>
    struct SampleValues<int>
    {
        static int first() { return 42; }
        static int second() { return -7; }
        static int zero() { return 0; }
    };

    template <>
    struct SampleValues<double>
    {
        static double first() { return 3.14; }
        static double second() { return -2.5; }
        static double zero() { return 0.0; }
    };

    template <>
    struct SampleValues<bool>
    {
        static bool first() { return true; }
        static bool second() { return false; }
        static bool zero() { return false; }
    };

    template <>
    struct SampleValues<std::string>
    {
        static std::string first() { return "hello"; }
        static std::string second() { return "world"; }
        static std::string zero() { return ""; }
    };

    template <>
    struct SampleValues<Color>
    {
        static Color first() { return Color::GREEN; }
        static Color second() { return Color::BLUE; }
        static Color zero() { return Color::RED; }
    };

    template <typename T>
    class SettingTypedTest : public ::testing::Test
    {
    };

    using ValueTypes = ::testing::Types<int, double, bool, std::string, Color>;
    TYPED_TEST_SUITE(SettingTypedTest, ValueTypes);

    TYPED_TEST(SettingTypedTest, notSetByDefault)
    {
        const settings::Setting<TypeParam> setting;

        EXPECT_FALSE(setting.isSet());
    }

    TYPED_TEST(SettingTypedTest, isSetAfterSet)
    {
        settings::Setting<TypeParam> setting;

        setting.set(SampleValues<TypeParam>::first());

        EXPECT_TRUE(setting.isSet());
    }

    TYPED_TEST(SettingTypedTest, getReturnsSetValue)
    {
        settings::Setting<TypeParam> setting;

        setting.set(SampleValues<TypeParam>::first());

        EXPECT_EQ(setting.get(), SampleValues<TypeParam>::first());
    }

    TYPED_TEST(SettingTypedTest, setOverwritesPreviousValue)
    {
        settings::Setting<TypeParam> setting;

        setting.set(SampleValues<TypeParam>::first());
        setting.set(SampleValues<TypeParam>::second());

        EXPECT_TRUE(setting.isSet());
        EXPECT_EQ(setting.get(), SampleValues<TypeParam>::second());
    }

    // isSet() tracks "was set", not "differs from the type's zero value"
    TYPED_TEST(SettingTypedTest, settingZeroValueStillCountsAsSet)
    {
        settings::Setting<TypeParam> setting;

        setting.set(SampleValues<TypeParam>::zero());

        EXPECT_TRUE(setting.isSet());
        EXPECT_EQ(setting.get(), SampleValues<TypeParam>::zero());
    }

    TYPED_TEST(SettingTypedTest, settingSameValueTwiceKeepsIsSet)
    {
        settings::Setting<TypeParam> setting;

        setting.set(SampleValues<TypeParam>::first());
        setting.set(SampleValues<TypeParam>::first());

        EXPECT_TRUE(setting.isSet());
        EXPECT_EQ(setting.get(), SampleValues<TypeParam>::first());
    }

    TYPED_TEST(SettingTypedTest, copyPreservesStateAndValue)
    {
        settings::Setting<TypeParam> original;
        original.set(SampleValues<TypeParam>::first());

        const auto copy = original;

        EXPECT_TRUE(copy.isSet());
        EXPECT_EQ(copy.get(), SampleValues<TypeParam>::first());
    }

    TYPED_TEST(SettingTypedTest, copyOfUnsetSettingIsUnset)
    {
        const settings::Setting<TypeParam> original;

        const auto copy = original;

        EXPECT_FALSE(copy.isSet());
    }

    TYPED_TEST(SettingTypedTest, copiesAreIndependent)
    {
        settings::Setting<TypeParam> original;
        original.set(SampleValues<TypeParam>::first());

        auto copy = original;
        copy.set(SampleValues<TypeParam>::second());

        EXPECT_EQ(original.get(), SampleValues<TypeParam>::first());
        EXPECT_EQ(copy.get(), SampleValues<TypeParam>::second());
    }

    // ---------------------------------------------------------------------
    // Default value behavior
    // ---------------------------------------------------------------------

    TYPED_TEST(SettingTypedTest, constructedWithDefaultIsNotSet)
    {
        const settings::Setting<TypeParam> setting(
            SampleValues<TypeParam>::first()
        );

        EXPECT_FALSE(setting.isSet());
    }

    TYPED_TEST(SettingTypedTest, getReturnsDefaultWhenNotSet)
    {
        const settings::Setting<TypeParam> setting(
            SampleValues<TypeParam>::first()
        );

        EXPECT_EQ(setting.get(), SampleValues<TypeParam>::first());
    }

    TYPED_TEST(SettingTypedTest, getDoesNotThrowWhenOnlyDefaultAvailable)
    {
        const settings::Setting<TypeParam> setting(
            SampleValues<TypeParam>::first()
        );

        EXPECT_NO_THROW(static_cast<void>(setting.get()));
    }

    TYPED_TEST(SettingTypedTest, setValueTakesPrecedenceOverDefault)
    {
        settings::Setting<TypeParam> setting(SampleValues<TypeParam>::first());

        setting.set(SampleValues<TypeParam>::second());

        EXPECT_TRUE(setting.isSet());
        EXPECT_EQ(setting.get(), SampleValues<TypeParam>::second());
    }

    // isSet() tracks "was explicitly set", even if the value equals the default
    TYPED_TEST(SettingTypedTest, settingValueEqualToDefaultCountsAsSet)
    {
        settings::Setting<TypeParam> setting(SampleValues<TypeParam>::first());

        setting.set(SampleValues<TypeParam>::first());

        EXPECT_TRUE(setting.isSet());
        EXPECT_EQ(setting.get(), SampleValues<TypeParam>::first());
    }

    TYPED_TEST(SettingTypedTest, copyPreservesDefault)
    {
        const settings::Setting<TypeParam> original(
            SampleValues<TypeParam>::first()
        );

        const auto copy = original;

        EXPECT_FALSE(copy.isSet());
        EXPECT_EQ(copy.get(), SampleValues<TypeParam>::first());
    }

    TYPED_TEST(SettingTypedTest, copyOfSetSettingWithDefaultKeepsSetValue)
    {
        settings::Setting<TypeParam> original(SampleValues<TypeParam>::first());
        original.set(SampleValues<TypeParam>::second());

        const auto copy = original;

        EXPECT_TRUE(copy.isSet());
        EXPECT_EQ(copy.get(), SampleValues<TypeParam>::second());
    }

    // ---------------------------------------------------------------------
    // No value and no default -> throws
    // ---------------------------------------------------------------------

    TYPED_TEST(SettingTypedTest, getWithoutValueAndWithoutDefaultThrows)
    {
        const settings::Setting<TypeParam> setting;

        EXPECT_THROW_MSG(
            static_cast<void>(setting.get()),
            exc::SettingsException,
            "Setting value is not set and no default is available."
        );
    }

    TYPED_TEST(SettingTypedTest, getDoesNotThrowOnceValueIsSet)
    {
        settings::Setting<TypeParam> setting;

        setting.set(SampleValues<TypeParam>::first());

        EXPECT_NO_THROW(static_cast<void>(setting.get()));
    }

    TYPED_TEST(SettingTypedTest, copyOfSettingWithoutValueAndDefaultThrows)
    {
        const settings::Setting<TypeParam> original;

        const auto copy = original;

        EXPECT_THROW_MSG(
            static_cast<void>(copy.get()),
            exc::SettingsException,
            "Setting value is not set and no default is available."
        );
    }

    // ---------------------------------------------------------------------
    // API / const-correctness (compile-time)
    // ---------------------------------------------------------------------

    TEST(SettingTest, getReturnsConstReference)
    {
        static_assert(
            std::is_same_v<
                decltype(std::declval<const settings::Setting<int>&>().get()),
                const int&>
        );
        SUCCEED();
    }

    TEST(SettingTest, readAccessorsCallableOnConstObject)
    {
        settings::Setting<int> setting;
        setting.set(5);

        const auto& constRef = setting;

        EXPECT_TRUE(constRef.isSet());
        EXPECT_EQ(constRef.get(), 5);
    }

    TEST(SettingTest, getReferenceTracksLaterSet)
    {
        settings::Setting<int> setting;
        setting.set(1);

        const int& ref = setting.get();
        setting.set(2);

        // reference to internal storage stays valid and reflects the update
        EXPECT_EQ(ref, 2);
    }

    // ---------------------------------------------------------------------
    // Settings are independent of each other
    // ---------------------------------------------------------------------

    TEST(SettingTest, differentInstancesAreIndependent)
    {
        settings::Setting<int> settingsA;
        settings::Setting<int> settingsB;

        settingsA.set(10);

        EXPECT_TRUE(settingsA.isSet());
        EXPECT_FALSE(settingsB.isSet());
    }
}   // namespace
