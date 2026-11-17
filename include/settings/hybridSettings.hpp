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
#include "setting.hpp"

/**
 * @class HybridSettings
 *
 * @brief stores all information about the external qmmm runner
 *
 */
class HybridSettings
{
   private:
    Setting<std::vector<size_t>> _innerRegionCenter;
    Setting<std::vector<int>>    _forcedCoreList;
    Setting<std::vector<int>>    _forcedLayerList;
    Setting<std::vector<int>>    _forcedOuterList;
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

        Setting<bool> _useQMCharges;

        Setting<double> _coreRadius;
        Setting<double> _layerRadius;
        Setting<double> _smoothingRegionThickness;
        Setting<double> _pointChargeThickness;

        Setting<SmoothingMethod>      _smoothing;
        Setting<QMForceDist>          _qmForceDist;
        static inline SmoothingMethod _smoothing   = SmoothingMethod::HOTSPOT;
        static inline QMForceDist     _qmForceDist = QMForceDist::EQUAL;

       public:
        HybridSettings();

        /********************
         * standard setters *
         ********************/

        void        setInnerRegionCenter(const std::vector<size_t> &);
        static void setInnerRegionCenterExpression(const std::string &);
        void        setForcedCoreList(const std::vector<int> &);
        void        setForcedLayerList(const std::vector<int> &);
        void        setForcedOuterList(const std::vector<int> &);

        void setUseQMCharges(bool useQMCharges);

        void setCoreRadius(double radius);
        void setLayerRadius(double radius);
        void setSmoothingRegionThickness(double thickness);
        void setPointChargeThickness(double radius);

        void setSmoothingMethod(SmoothingMethod method);
        void setQMForceDist(QMForceDist method);

        /**
         * @brief resolves pending selection expressions (needs the files)
         */
        static void finalize(const FileSettings &fileSettings);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] std::optional<std::vector<size_t>> getInnerRegionCenter(
        ) const;
        [[nodiscard]] const std::vector<int> &getForcedCoreList() const;
        [[nodiscard]] const std::vector<int> &getForcedLayerList() const;
        [[nodiscard]] const std::vector<int> &getForcedOuterList() const;

        [[nodiscard]] bool getUseQMCharges() const;

        [[nodiscard]] double getCoreRadius() const;
        [[nodiscard]] double getLayerRadius() const;
        [[nodiscard]] double getSmoothingRegionThickness() const;
        [[nodiscard]] double getPointChargeThickness() const;

        [[nodiscard]] SmoothingMethod getSmoothingMethod() const;
        [[nodiscard]] QMForceDist     getQMForceDist() const;
    };

#endif   // SETTINGS__HYBRID_SETTINGS_HPP
