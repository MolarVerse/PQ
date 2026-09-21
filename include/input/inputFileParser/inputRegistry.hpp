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

#ifndef _INPUT_REGISTRY_HPP_
#define _INPUT_REGISTRY_HPP_

#include "inputParam.hpp"

namespace input
{
    /**
     * @class InputRegistry
     *
     * @brief owns a set of InputKey<T> instances for one InputFileParser
     * subclass: register once with defaults, parse lines, query later
     *
     */
    class InputRegistry
    {
       private:
        std::unordered_map<std::string, std::unique_ptr<InputKeyBase>> _keys;

       public:
        template <typename T>
        InputKey<T> &registerKey(const KeyRegistry<T> &keyRegistry);

        void parseLine(
            const std::vector<std::string> &lineElements,
            size_t                          lineNumber
        );

        template <typename T>
        [[nodiscard]]
        const InputKey<T> &get(const std::string &name) const;

        [[nodiscard]]
        std::vector<std::string> describeAll() const;

        void clearValues();
    };
}   // namespace input

#ifndef _INPUT_REGISTRY_TPP_
#include "inputRegistry.tpp"
#endif

#endif   // _INPUT_REGISTRY_HPP_
