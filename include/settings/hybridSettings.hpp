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
#include <vector>        // for vector

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
        static inline std::vector<int> _innerRegionCenter;
        static inline std::vector<int> _forcedInnerList;
        static inline std::vector<int> _forcedOuterList;

        static inline bool _useQMCharges = false;

        static inline double _coreRadius               = 0.0;
        static inline double _layerRadius              = 0.0;
        static inline double _smoothingRegionThickness = 0.0;
        static inline double _pointChargeThickness     = 0.0;

       public:
        /********************
         * standard setters *
         ********************/

        static void setInnerRegionCenter(const std::vector<int> &);
        static void setForcedInnerList(const std::vector<int> &);
        static void setForcedOuterList(const std::vector<int> &);

        static void setUseQMCharges(const bool useQMCharges);

        static void setCoreRadius(const double radius);
        static void setLayerRadius(const double radius);
        static void setSmoothingRegionThickness(const double thickness);
        static void setPointChargeThickness(const double radius);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::vector<int> getInnerRegionCenter();
        [[nodiscard]] static std::vector<int> getForcedInnerList();
        [[nodiscard]] static std::vector<int> getForcedOuterList();

        [[nodiscard]] static bool getUseQMCharges();

        [[nodiscard]] static double getCoreRadius();
        [[nodiscard]] static double getLayerRadius();
        [[nodiscard]] static double getSmoothingRegionThickness();
        [[nodiscard]] static double getPointChargeThickness();
    };
}   // namespace settings

#endif   // _HYBRID_SETTINGS_HPP_