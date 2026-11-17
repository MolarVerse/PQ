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

#ifndef SETTINGS__SETTING_TPP
#define SETTINGS__SETTING_TPP

#include "exceptions.hpp"
#include "setting.hpp"

/**
 * Construct a setting with a default value.
 *
 * @param defaultValue the default value for the setting.
 */
template <typename T, mstd::Tag Name>
Setting<T, Name>::Setting(const T& defaultValue)
    : _default(defaultValue), _setter(Setter::Default)
{
}

/**
 * Check if the setting has been set by the user.
 *
 * @return true if the setting has a value set by the user, false otherwise.
 */
template <typename T, mstd::Tag Name>
bool Setting<T, Name>::isSetByUser() const
{
    return _value.has_value() && _setter == Setter::User;
}

/**
 * Check if the setting has been set by the system.
 *
 * @return true if the setting has a value set by the system, false otherwise.
 */
template <typename T, mstd::Tag Name>
bool Setting<T, Name>::isSetBySystem() const
{
    return _value.has_value() && _setter == Setter::System;
}

/**
 * Check if the setting has been set by anyone (user or system).
 *
 * @return true if the setting has a value set by either the user or the system,
 * false otherwise.
 */
template <typename T, mstd::Tag Name>
bool Setting<T, Name>::isSetBySomeone() const
{
    return _value.has_value() && _setter != Setter::None;
}

/**
 * Get the value of the setting.
 *
 * @return the value of the setting if set, otherwise the default value.
 * @throws exc::SettingsException if the setting is not set and no default
 * is available.
 */
template <typename T, mstd::Tag Name>
const T& Setting<T, Name>::get() const
{
    if (!_value.has_value() && !_default.has_value())
        _throwIfNotSet();

    if (!_value.has_value())
        return _default.value();

    return _value.value();
}

/**
 * Set the value of the setting.
 *
 * @param value the value to set for the setting.
 */
template <typename T, mstd::Tag Name>
void Setting<T, Name>::setByUser(const T& value)
{
    _value  = value;
    _setter = Setter::User;
}

/**
 * @brief Set the value of the setting by the system.
 *
 * @tparam T
 * @tparam Name
 * @param value
 */
template <typename T, mstd::Tag Name>
void Setting<T, Name>::setBySystem(const T& value)
{
    _value  = value;
    _setter = Setter::System;
}

/**
 * Throw an exception if the setting is not set and no default is available.
 *
 * @throws exc::SettingsException if the setting is not set and no default
 * is available.
 */
template <typename T, mstd::Tag Name>
void Setting<T, Name>::_throwIfNotSet() const
{
    std::string errorMessage =
        "Setting value is not set and no default is available.";

    errorMessage = Name.to_string() + ": " + errorMessage;

    throw exc::SettingsException(errorMessage);
}

#endif   // SETTINGS__SETTING_TPP
