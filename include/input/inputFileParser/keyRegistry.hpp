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

#ifndef _KEY_REGISTRY_HPP_
#define _KEY_REGISTRY_HPP_

#include "keyMetaData.hpp"
#include "keyValidatorBase.hpp"

namespace input
{
    template <typename T>
    struct KeyRegistry
    {
        using CustomParser = std::function<std::optional<T>(std::string_view)>;

        KeyMetadata                    metadata;
        std::optional<T>               defaultValue = std::nullopt;
        std::optional<std::vector<T>>  allowed      = std::nullopt;
        CustomParser                   customParser = nullptr;
        std::function<void(const T &)> onSet        = nullptr;

        // shared + const: validators are immutable, so sharing one instance
        // between the registry description and the key is safe and keeps
        // KeyRegistry copyable
        std::shared_ptr<const KeyValidator<T>> validator = nullptr;
    };
}   // namespace input

#endif   // _KEY_REGISTRY_HPP_
