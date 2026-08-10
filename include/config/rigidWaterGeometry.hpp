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

#ifndef _RIGID_WATER_GEOMETRY_HPP_

#define _RIGID_WATER_GEOMETRY_HPP_

namespace constants
{
    static constexpr double SPC_OH_DIST = 1.0;           // Angström
    static constexpr double SPC_HH_DIST = 1.632993162;   // Angström

    static constexpr double SPC_E_OH_DIST = SPC_OH_DIST;   // Angström
    static constexpr double SPC_E_HH_DIST = SPC_HH_DIST;   // Angström

    static constexpr double SPC_DC_OH_DIST = SPC_OH_DIST;   // Angström
    static constexpr double SPC_DC_HH_DIST = SPC_HH_DIST;   // Angström

    static constexpr double H2O_DC_OH_DIST = 0.958;     // Angström
    static constexpr double H2O_DC_HH_DIST = 1.56441;   // Angström

    static constexpr double TIP3P_OH_DIST = 0.9572;   // Angström
    static constexpr double TIP3P_HH_DIST = 1.5139;   // Angström

    static constexpr double OPC3_OH_DIST = 0.97888;       // Angström
    static constexpr double OPC3_HH_DIST = 1.598492306;   // Angström

}   // namespace constants

#endif   // _RIGID_WATER_GEOMETRY_HPP_
