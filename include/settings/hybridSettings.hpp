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

#include <cstdint>
#include <optional>   // for optional
#include <string>     // for string
#include <vector>     // for vector

namespace settings
{
    /**
     * @enum SmoothingMethod
     *
     * @brief enum class to store the type of smoothing method
     *
     */
    enum class SmoothingMethod : std::uint8_t
    {
        HOTSPOT,
        EXACT
    };

    /**
     * @enum QMForceDist
     *
     * @brief enum class to store the type of force distribution of the QM
     * method in hotspot smoothing
     *
     */
    enum class QMForceDist : std::uint8_t
    {
        NONE,
        EQUAL,
        RANDOM,
        DISTANCE_WEIGHTED
    };

    [[nodiscard]]
    std::string string(SmoothingMethod method);

    /**
     * @class HybridSettings
     *
     * @brief stores all information about the external qmmm runner
     *
     */
    class HybridSettings
    {
       private:
        static inline std::optional<std::vector<size_t>> _innerRegionCenter;
        static inline std::vector<int>                   _forcedCoreList;
        static inline std::vector<int>                   _forcedLayerList;
        static inline std::vector<int>                   _forcedOuterList;

        static inline bool _useQMCharges = true;

        static inline double _coreRadius               = 0.0;
        static inline double _layerRadius              = 0.0;
        static inline double _smoothingRegionThickness = 0.0;
        static inline double _pointChargeThickness     = 0.0;

        static inline SmoothingMethod _smoothing   = SmoothingMethod::HOTSPOT;
        static inline QMForceDist     _qmForceDist = QMForceDist::NONE;

       public:
        /********************
         * standard setters *
         ********************/

        static void setInnerRegionCenter(const std::vector<size_t> &);
        static void setForcedCoreList(const std::vector<int> &);
        static void setForcedLayerList(const std::vector<int> &);
        static void setForcedOuterList(const std::vector<int> &);

        static void setUseQMCharges(bool useQMCharges);

        static void setCoreRadius(double radius);
        static void setLayerRadius(double radius);
        static void setSmoothingRegionThickness(double thickness);
        static void setPointChargeThickness(double radius);

        static void setSmoothingMethod(SmoothingMethod method);
        static void setQMForceDist(QMForceDist method);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::optional<std::vector<size_t>> getInnerRegionCenter(
        );
        [[nodiscard]] static std::vector<int> getForcedCoreList();
        [[nodiscard]] static std::vector<int> getForcedLayerList();
        [[nodiscard]] static std::vector<int> getForcedOuterList();

        [[nodiscard]] static bool getUseQMCharges();

        [[nodiscard]] static double getCoreRadius();
        [[nodiscard]] static double getLayerRadius();
        [[nodiscard]] static double getSmoothingRegionThickness();
        [[nodiscard]] static double getPointChargeThickness();

        [[nodiscard]] static SmoothingMethod getSmoothingMethod();
        [[nodiscard]] static QMForceDist     getQMForceDist();
    };
}   // namespace settings

#endif   // _HYBRID_SETTINGS_HPP_
