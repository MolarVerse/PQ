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

namespace settings
{
    /**
     * @class ForceFieldSettings
     *
     * @brief static class to store settings of the force field
     *
     */
    class ForceFieldSettings
    {
      private:
        static inline bool _active = false;

      public:
        ForceFieldSettings()  = default;
        ~ForceFieldSettings() = default;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static bool isActive() { return _active; }

        /********************
         * standard setters *
         ********************/

        static void activate() { _active = true; }
        static void deactivate() { _active = false; }
    };

}   // namespace settings

#endif   // _FORCE_FIELD_SETTINGS_HPP_