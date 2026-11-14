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

#ifndef SETTINGS__HYBRID_SETTINGS_HPP
#define SETTINGS__HYBRID_SETTINGS_HPP

#include <optional>
#include <string>
#include <vector>

#include "enums/hybrid.hpp"
#include "fileSettings.hpp"

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
        static inline std::optional<std::vector<size_t>> _innerRegionCenter;
        static inline std::optional<std::string> _innerRegionCenterExpression;
        static inline std::vector<int>           _forcedCoreList;
        static inline std::vector<int>           _forcedLayerList;
        static inline std::vector<int>           _forcedOuterList;

        static inline bool _useQMCharges = true;

        static inline Distance _coreRadius               = Distance::zero();
        static inline Distance _layerRadius              = Distance::zero();
        static inline Distance _smoothingRegionThickness = Distance::zero();
        static inline Distance _pointChargeThickness     = Distance::zero();

        static inline SmoothingMethod _smoothing   = SmoothingMethod::HOTSPOT;
        static inline QMForceDist     _qmForceDist = QMForceDist::EQUAL;

       public:
        /********************
         * standard setters *
         ********************/

        static void setInnerRegionCenter(const std::vector<size_t> &);
        static void setInnerRegionCenterExpression(const std::string &);
        static void setForcedCoreList(const std::vector<int> &);
        static void setForcedLayerList(const std::vector<int> &);
        static void setForcedOuterList(const std::vector<int> &);

        static void setUseQMCharges(bool useQMCharges);

        static void setCoreRadius(const Distance &radius);
        static void setLayerRadius(const Distance &radius);
        static void setSmoothingRegionThickness(const Distance &thickness);
        static void setPointChargeThickness(const Distance &radius);

        static void setSmoothingMethod(SmoothingMethod method);
        static void setQMForceDist(QMForceDist method);

        /**
         * @brief resolves pending selection expressions (needs the files)
         */
        static void finalize(const FileSettings &fileSettings);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::optional<std::vector<size_t>> getInnerRegionCenter(
        );
        [[nodiscard]] static std::vector<int> getForcedCoreList();
        [[nodiscard]] static std::vector<int> getForcedLayerList();
        [[nodiscard]] static std::vector<int> getForcedOuterList();

        [[nodiscard]] static bool getUseQMCharges();

        [[nodiscard]] static Distance getCoreRadius();
        [[nodiscard]] static Distance getLayerRadius();
        [[nodiscard]] static Distance getSmoothingRegionThickness();
        [[nodiscard]] static Distance getPointChargeThickness();

        [[nodiscard]] static SmoothingMethod getSmoothingMethod();
        [[nodiscard]] static QMForceDist     getQMForceDist();
    };
}   // namespace settings

#endif   // SETTINGS__HYBRID_SETTINGS_HPP
