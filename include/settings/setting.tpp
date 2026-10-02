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

#ifndef _SETTING_TPP_
#define _SETTING_TPP_

#include "exceptions.hpp"
#include "setting.hpp"

namespace settings
{
    /**
     * Check if the setting has been set.
     *
     * @return true if the setting has a value, false otherwise.
     */
    template <typename T>
    bool Setting<T>::isSet() const
    {
        return _value.has_value();
    }

    /**
     * Get the value of the setting.
     *
     * @return the value of the setting if set, otherwise the default value.
     * @throws exc::SettingsException if the setting is not set and no default
     * is available.
     */
    template <typename T>
    const T& Setting<T>::get() const
    {
        if (!_value.has_value() && !_default.has_value())
            throw exc::SettingsException(
                "Setting value is not set and no default is available."
            );

        if (!_value.has_value())
            return _default.value();

        return _value.value();
    }

    /**
     * Set the value of the setting.
     *
     * @param value the value to set for the setting.
     */
    template <typename T>
    void Setting<T>::set(const T& value)
    {
        _value = value;
    }
}   // namespace settings

#endif   // _SETTING_TPP_
