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

#ifndef _FORCE_FIELD_SETTINGS_HPP_

#define _FORCE_FIELD_SETTINGS_HPP_

#include "enums/potential.hpp"
#include "setting.hpp"

/**
 * @class ForceFieldSettings
 *
 * @brief static class to store settings of the force field
 *
 */
class ForceFieldSettings
{
   private:
    Setting<ForceFieldType> _type;

   public:
    ForceFieldSettings();

    /********************
     * standard getters *
     ********************/

    [[nodiscard]] bool           isActive() const;
    [[nodiscard]] bool           isNonCoulombicActive() const;
    [[nodiscard]] ForceFieldType getType() const;

    /********************
     * standard setters *
     ********************/

    void setType(ForceFieldType value);
};

#endif   // _FORCE_FIELD_SETTINGS_HPP_
