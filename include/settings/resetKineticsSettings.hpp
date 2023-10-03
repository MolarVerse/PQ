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

#ifndef _RESET_KINETICS_SETTINGS_HPP_

#define _RESET_KINETICS_SETTINGS_HPP_

#include <cstddef>   // for size_t

namespace settings
{
    /**
     * @class ResetKineticsSettings
     *
     * @brief static class to store settings of reset kinetics
     *
     */
    class ResetKineticsSettings
    {
      private:
        static inline size_t _nScale        = 0;
        static inline size_t _fScale        = 0;
        static inline size_t _nReset        = 0;
        static inline size_t _fReset        = 0;
        static inline size_t _nResetAngular = 0;
        static inline size_t _fResetAngular = 0;

      public:
        ResetKineticsSettings()  = default;
        ~ResetKineticsSettings() = default;

        /***************************
         * standard setter methods *
         ***************************/

        static void setNScale(const size_t nScale) { _nScale = nScale; }
        static void setFScale(const size_t fScale) { _fScale = fScale; }
        static void setNReset(const size_t nReset) { _nReset = nReset; }
        static void setFReset(const size_t fReset) { _fReset = fReset; }
        static void setNResetAngular(const size_t nResetAngular) { _nResetAngular = nResetAngular; }
        static void setFResetAngular(const size_t fResetAngular) { _fResetAngular = fResetAngular; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static size_t getNScale() { return _nScale; }
        [[nodiscard]] static size_t getFScale() { return _fScale; }
        [[nodiscard]] static size_t getNReset() { return _nReset; }
        [[nodiscard]] static size_t getFReset() { return _fReset; }
        [[nodiscard]] static size_t getNResetAngular() { return _nResetAngular; }
        [[nodiscard]] static size_t getFResetAngular() { return _fResetAngular; }
    };
}   // namespace settings

#endif   // _RESET_KINETICS_SETTINGS_HPP_