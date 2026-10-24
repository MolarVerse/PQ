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

#include "forceFieldSettings.hpp"

#include <utility>

namespace settings
{

    /**
     * @brief Get the type of the force field
     *
     * @return ForceFieldType
     */
    ForceFieldType ForceFieldSettings::getType() { return _type; }

    /**
     * @brief Set the type of the force field
     *
     * @param value
     */
    void ForceFieldSettings::setType(ForceFieldType value) { _type = value; }

    /**
     * @brief Get if the force field is active
     *
     * @return ForceFieldType
     */
    bool ForceFieldSettings::isActive()
    {
        switch (_type)
        {
            case ForceFieldType::OFF: return false;
            case ForceFieldType::ON:
            case ForceFieldType::BONDED: return true;
        }

        std::unreachable();
    }

    /**
     * @brief Get if the non-Coulombic part of the force field is active
     *
     * @return bool
     */
    bool ForceFieldSettings::isNonCoulombicActive()
    {
        switch (_type)
        {
            case ForceFieldType::OFF:
            case ForceFieldType::BONDED: return false;
            case ForceFieldType::ON: return true;
        }

        std::unreachable();
    }

}   // namespace settings
