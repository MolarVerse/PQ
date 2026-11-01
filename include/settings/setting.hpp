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

#ifndef SETTINGS__SETTING_HPP
#define SETTINGS__SETTING_HPP

#include <mstd/types.hpp>
#include <optional>

/**
 * @brief A template class representing a configurable setting.
 *
 * @tparam T The type of the setting value.
 * @tparam Tag The tag type providing a name for the setting. This should be an
 * instance of `mstd::Tag` with a unique string identifier. The tag is used to
 * generate meaningful error messages when the setting is not set and no default
 * is available.
 */
template <typename T, mstd::Tag Name = mstd::Tag("Setting")>
class Setting
{
   private:
    std::optional<T> _default = std::nullopt;
    std::optional<T> _value   = std::nullopt;

   public:
    explicit Setting() = default;
    explicit Setting(const T& defaultValue);

    [[nodiscard]]
    const T& get() const;

    void set(const T& value);

    [[nodiscard]]
    bool isSet() const;
};

#ifndef SETTINGS__SETTING_TPP
#include "setting.tpp"
#endif

#endif   // SETTINGS__SETTING_HPP
