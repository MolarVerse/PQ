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

#ifndef TESTS__INCLUDE__TEST_UTILS__TEST_SETTINGS_HPP
#define TESTS__INCLUDE__TEST_UTILS__TEST_SETTINGS_HPP

#include <utility>

/**
 * @brief Helpers to construct settings objects in tests without the input file
 * parser and without the static settings facade.
 *
 * @details Every settings struct is default-constructible, so a test builds
 * the one it needs, overrides what it cares about and hands it to the
 * consumer by const reference:
 *
 * @code
 * const auto settings =
 *     testSettings::make<ConstraintSettings>()
 *         .with(&ConstraintSettings::setShakeMaxIter, 100)
 *         .with(&ConstraintSettings::setShakeTolerance, 1e-8)
 *         .apply([](auto &config) { config.activateShake(); })
 *         .build();
 * constraints::Constraints constraints(settings);
 * @endcode
 *
 * Consumers keep a reference to the settings they are given, so keep the
 * result of build() in a named variable that outlives the consumer; never pass
 * a temporary.
 *
 * Values that were not overridden keep the defaults of the default
 * constructor. Settings without a sensible default (for example the water
 * model) throw on access until they are set.
 */
namespace testSettings
{
    /**
     * @class Builder
     *
     * @brief Collects overrides for one settings struct.
     *
     * @tparam SettingsT the settings struct, for example ConstraintSettings or
     * the root Settings
     */
    template <typename SettingsT>
    class Builder
    {
       private:
        SettingsT _settings;

       public:
        /**
         * @brief set one value through its setter
         *
         * @param setter pointer to a setter taking one argument, for example
         * &ConstraintSettings::setShakeMaxIter
         * @param value the value, converted to the parameter type of the setter
         */
        template <typename Class, typename Argument, typename Value>
        Builder &with(void (Class::*setter)(Argument), Value &&value)
        {
            (_settings.*
             setter)(static_cast<Argument>(std::forward<Value>(value)));
            return *this;
        }

        /**
         * @brief change the settings with any callable, for everything that
         * is not a plain setter (activate/deactivate, nested structs)
         *
         * @param callable called with a reference to the settings struct
         */
        template <typename Callable>
        Builder &apply(Callable &&callable)
        {
            std::forward<Callable>(callable)(_settings);
            return *this;
        }

        /**
         * @brief the settings built so far (a copy, the builder can be reused)
         */
        [[nodiscard]]
        SettingsT build() const
        {
            return _settings;
        }
    };

    /**
     * @brief start a builder for the settings struct SettingsT
     */
    template <typename SettingsT>
    [[nodiscard]]
    Builder<SettingsT> make()
    {
        return Builder<SettingsT>{};
    }
}   // namespace testSettings

#endif   // TESTS__INCLUDE__TEST_UTILS__TEST_SETTINGS_HPP
