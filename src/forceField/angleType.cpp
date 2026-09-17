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

#include "angleType.hpp"

namespace forceField
{

    /**
     * @brief Construct a new Angle Type:: Angle Type object
     *
     * @param id
     * @param params
     */
    AngleType::AngleType(const AngleId id, const AngleParams &params)
        : _id(id), _params(params)
    {
    }

    /**
     * @brief operator overload for the comparison of two AngleType objects
     *
     * @param other
     * @return true
     * @return false
     */
    bool operator==(const AngleType &self, const AngleType &other)
    {
        return self._id == other._id && self._params == other._params;
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the id of the angle type
     *
     * @return AngleId
     */
    AngleId AngleType::getId() const { return _id; }

    /**
     * @brief get the parameters of the angle type
     *
     * @return const AngleParams&
     */
    const AngleParams &AngleType::getParams() const { return _params; }

}   // namespace forceField
