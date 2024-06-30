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

        static inline double _qmCoreRadius        = 0.0;
        static inline double _qmmmLayerRadius     = 0.0;
        static inline double _qmmmSmoothingRadius = 0.0;

       public:
        /********************
         * standard setters *
         ********************/

        static void setQMCenterString(const std::string_view qmCenter);
        static void setQMOnlyListString(const std::string_view qmOnlyList);
        static void setMMOnlyListString(const std::string_view mmOnlyList);

        static void setUseQMCharges(const bool useQMCharges);

        static void setQMCoreRadius(const double qmCoreRadius);
        static void setQMMMLayerRadius(const double qmmmLayerRadius);
        static void setQMMMSmoothingRadius(const double qmmmSmoothingRadius);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::string getQMCenterString();
        [[nodiscard]] static std::string getQMOnlyListString();
        [[nodiscard]] static std::string getMMOnlyListString();

        [[nodiscard]] static bool getUseQMCharges();

        [[nodiscard]] static double getQMCoreRadius();
        [[nodiscard]] static double getQMMMLayerRadius();
        [[nodiscard]] static double getQMMMSmoothingRadius();
    };
}   // namespace settings

#endif   // _QMMM_SETTINGS_HPP_