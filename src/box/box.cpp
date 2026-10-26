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

#include "box.hpp"

#include "units.hpp"

namespace molsys
{
    /**
     * @brief Construct a new Box object
     *
     * @param boxDimensions
     * @param angles
     */
    Box::Box(const linalg::Vec3D &boxDimensions, const linalg::Vec3D &angles)
        : _boxDimensions(boxDimensions), _angles((angles * deg).in(rad))
    {
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the box angles
     *
     * @return linalg::Vec3D
     */
    linalg::Vec3D Box::getBoxAngles() const
    {
        return _angles.get().numerical_value_in(deg);
    }

}   // namespace molsys
