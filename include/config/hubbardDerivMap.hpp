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

#ifndef _HUBBARD_DERIV_MAP_HPP_

#define _HUBBARD_DERIV_MAP_HPP_

#include <unordered_map>   // for unordered_map
#include <string>

namespace constants
{
    /**
     * @brief Map of Hubbard Derivatives for the 3OB Parameter Set
     */
    const std::unordered_map<std::string, double> hubbardDerivMap3ob = {
        {"H", -0.1857}, {"S", -0.11}, {"P", -0.14}, {"F", -0.1623}, {"Cl", -0.0697},
        {"Br", -0.0573}, {"I", -0.0433}, {"Zn", -0.03}, {"Mg", -0.02}, {"Ca", -0.0340},
        {"K", -0.0339}, {"Na", -0.0454}, {"C", -0.1492}, {"N", -0.1535}, {"O", -0.1575}
    };

}   // namespace constants

#endif   // _HUBBARD_DERIV_MAP_HPP_
