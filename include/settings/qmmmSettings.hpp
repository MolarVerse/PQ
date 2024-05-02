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

#ifndef _QMMM_SETTINGS_HPP_

#define _QMMM_SETTINGS_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @class QMMMSettings
     *
     * @brief stores all information about the external qmmm runner
     *
     */
    class QMMMSettings
    {
       private:
        static inline std::string _qmCenterString   = "";
        static inline std::string _qmOnlyListString = "";
        static inline std::string _mmOnlyListString = "";

        static inline bool _useQMCharges = false;

        static inline double _qmCoreRadius = 0.0;

       public:
        /********************
         * standard setters *
         ********************/
        static void setQMCenterString(const std::string_view qmCenter)
        {
            _qmCenterString = qmCenter;
        }

        static void setQMOnlyListString(const std::string_view qmOnlyList)
        {
            _qmOnlyListString = qmOnlyList;
        }

        static void setMMOnlyListString(const std::string_view mmOnlyList)
        {
            _mmOnlyListString = mmOnlyList;
        }

        static void setUseQMCharges(const bool useQMCharges) { _useQMCharges = useQMCharges; }

        static void setQMCoreRadius(const double qmCoreRadius) { _qmCoreRadius = qmCoreRadius; }

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::string getQMCenterString() { return _qmCenterString; }
        [[nodiscard]] static std::string getQMOnlyListString() { return _qmOnlyListString; }
        [[nodiscard]] static std::string getMMOnlyListString() { return _mmOnlyListString; }
        [[nodiscard]] static bool        getUseQMCharges() { return _useQMCharges; }
        [[nodiscard]] static double      getQMCoreRadius() { return _qmCoreRadius; }
    };
}   // namespace settings

#endif   // _QMMM_SETTINGS_HPP_