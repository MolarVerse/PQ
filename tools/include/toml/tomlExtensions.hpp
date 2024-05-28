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

#ifndef _TOML_EXTENSIONS_HPP_

#define _TOML_EXTENSIONS_HPP_

#include "toml.hpp"
#include <vector>

namespace tomlExtensions
{
    template <typename T>
    std::vector<T> tomlArrayToVector(const toml::array *array)
    {
        std::vector<T> vec;
        array->for_each(
            [&vec](const toml::value<T> &element)
            {
                vec.push_back(element.value_or(T()));
            });
        return vec;
    }
}

#endif // _TOML_EXTENSIONS_HPP_