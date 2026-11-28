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

#include <cstdint>
#include <mstd/types.hpp>
#include <optional>

#define SETTING_DETAIL_CONST_GETTER(fn, member)                              \
    [[nodiscard]] decltype(auto) fn() const noexcept(noexcept(member.get())) \
    {                                                                        \
        return member.get();                                                 \
    }

#define SETTING_DETAIL_SETTER_BY_USER(fn, member)                             \
    void fn(                                                                  \
        ::mstd::detail::getter_t<std::remove_cvref_t<decltype(member.get())>> \
            value                                                             \
    ) noexcept(noexcept(member.setByUser(value)))                             \
    {                                                                         \
        member.setByUser(value);                                              \
    }

#define SETTING_GETTER(name) SETTING_DETAIL_CONST_GETTER(get##name, name)

#define SETTING_SETTER_BY_USER(name) \
    SETTING_DETAIL_SETTER_BY_USER(set##name##ByUser, name)

#define SETTING_GETTER_AND_SETTER_BY_USER(name) \
    SETTING_GETTER(name)                        \
    SETTING_SETTER_BY_USER(name)

/**
 * @brief A template class representing a configurable setting.
 *
 * @tparam T The type of the setting value.
 * @tparam Tag The tag type providing a name for the setting. This should be an
 * instance of `mstd::Tag` with a unique string identifier. The tag is used to
 * generate meaningful error messages when the setting is not set and no default
 * is available.
 */
template <typename T, mstd::Tag Name>
class Setting
{
   public:
    enum class Setter : std::uint8_t
    {
        None,
        Default,
        User,
        System
    };

   private:
    std::optional<T> _default = std::nullopt;
    std::optional<T> _value   = std::nullopt;
    Setter           _setter  = Setter::None;

   public:
    explicit Setting() = default;
    explicit Setting(const T& defaultValue);

    [[nodiscard]]
    const T& get() const;

    void setByUser(const T& value);
    void setBySystem(const T& value);

    [[nodiscard]]
    bool isSetByUser() const;
    [[nodiscard]]
    bool isSetBySystem() const;
    [[nodiscard]]
    bool isSetBySomeone() const;

   private:
    void _throwIfNotSet() const;
};

#ifndef SETTINGS__SETTING_TPP
#include "setting.tpp"
#endif

#endif   // SETTINGS__SETTING_HPP
