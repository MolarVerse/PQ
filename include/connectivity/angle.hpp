/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _ANGLE_HPP_

#define _ANGLE_HPP_

#include "connectivityElement.hpp"

namespace connectivity
{
    /**
     * @class Angle
     *
     * @brief Represents an angle between three atoms.
     *
     */
    class Angle : public ConnectivityElement
    {
      public:
        using ConnectivityElement::ConnectivityElement;
    };

}   // namespace connectivity

#endif   // _ANGLE_HPP_