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

#ifndef _RING_POLYMER_SETTINGS_HPP_

#define _RING_POLYMER_SETTINGS_HPP_

#include <cstddef>   // for size_t

namespace settings
{
    /**
     * @class RingPolymerSettings
     *
     * @brief class for storing settings for ring polymer md
     *
     */
    class RingPolymerSettings
    {
      private:
        static inline bool _numberOfBeadsSet = false;

        static inline size_t _numberOfBeads = 0;

      public:
        static void setNumberOfBeads(const size_t numberOfBeads);

        [[nodiscard]] static size_t getNumberOfBeads() { return _numberOfBeads; }
        [[nodiscard]] static bool   isNumberOfBeadsSet() { return _numberOfBeadsSet; }
    };
}   // namespace settings

#endif   // _RING_POLYMER_SETTINGS_HPP_