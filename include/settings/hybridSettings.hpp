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

#ifndef _HYBRID_SETTINGS_HPP_

#define _HYBRID_SETTINGS_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @class HybridSettings
     *
     * @brief stores all information about the external qmmm runner
     *
     */
    class HybridSettings
    {
       private:
        static inline std::string _coreCenterString      = "";
        static inline std::string _forcedInnerListString = "";
        static inline std::string _forcedOuterListString = "";

        static inline bool _useQMCharges = false;

        static inline double _coreRadius      = 0.0;
        static inline double _layerRadius     = 0.0;
        static inline double _smoothingRadius = 0.0;

       public:
        /********************
         * standard setters *
         ********************/

        static void setCoreCenterString(const std::string_view qmCenter);
        static void setForcedInnerListString(
            const std::string_view forcedInnerList
        );
        static void setForcedOuterListString(
            const std::string_view forcedOuterList
        );

        static void setUseQMCharges(const bool useQMCharges);

        static void setCoreRadius(const double qmCoreRadius);
        static void setLayerRadius(const double qmmmLayerRadius);
        static void setSmoothingRadius(const double qmmmSmoothingRadius);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::string getCoreCenterString();
        [[nodiscard]] static std::string getForcedInnerListString();
        [[nodiscard]] static std::string getForcedOuterListString();

        [[nodiscard]] static bool getUseQMCharges();

        [[nodiscard]] static double getCoreRadius();
        [[nodiscard]] static double getLayerRadius();
        [[nodiscard]] static double getSmoothingRadius();
    };
}   // namespace settings

#endif   // _HYBRID_SETTINGS_HPP_