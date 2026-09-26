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

#ifndef _INPUT_REGISTRY_TPP_
#define _INPUT_REGISTRY_TPP_

#include "inputRegistry.hpp"

namespace input
{
    /**
     * @brief Registers a new key in the input registry.
     *
     * @tparam T the type of the key
     * @param keyRegistry the key registry containing metadata and default value
     * @return a reference to the newly registered InputKey
     *
     * @throws std::logic_error if the key is already registered
     */
    template <typename T>
    InputKey<T> &InputRegistry::registerKey(const KeyRegistry<T> &keyRegistry)
    {
        const std::string name = keyRegistry.metadata.name;

        auto key = std::make_unique<InputKey<T>>(keyRegistry);

        auto [it, inserted] = _keys.try_emplace(name, std::move(key));
        if (!inserted)
            throw std::logic_error(
                std::format("Key \"{}\" registered twice", name)
            );
        return static_cast<InputKey<T> &>(*it->second);
    }

    /**
     * @brief retrieves a registered key by name
     *
     * @tparam T the type of the key
     * @param name the name of the key
     * @return a const reference to the requested InputKey
     * @throws std::out_of_range if the key is not found
     */
    template <typename T>
    const InputKey<T> &InputRegistry::get(const std::string &name) const
    {
        return dynamic_cast<const InputKey<T> &>(*_keys.at(name));
    }

}   // namespace input

#endif   // _INPUT_REGISTRY_TPP_
