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

#include "atomicNumber.hpp"

#include <string>

#include "atomNumberMap.hpp"
#include "stringUtilities.hpp"

namespace out
{
    /**
     * @brief atomic number of an atom for trajectory files, 0 for atoms
     * without an element (dummy and virtual atoms)
     *
     * @param name PQ atom name
     * @return int
     */
    int atomicNumber(const std::string &name)
    {
        const auto key = utilities::toLowerCopy(name);

        if (!atomNumberMap.contains(key) || nonElementAtomNames.contains(key))
        {
            return 0;
        }

        return static_cast<int>(atomNumberMap.at(key).get());
    }

}   // namespace out
