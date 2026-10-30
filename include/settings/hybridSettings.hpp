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

#include <optional>
#include <vector>

#include "enums/hybrid.hpp"
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

    Setting<bool> _useQMCharges;

    Setting<double> _coreRadius;
    Setting<double> _layerRadius;
    Setting<double> _smoothingRegionThickness;
    Setting<double> _pointChargeThickness;

    Setting<SmoothingMethod> _smoothing;
    Setting<QMForceDist>     _qmForceDist;

   public:
    HybridSettings();

    /********************
     * standard setters *
     ********************/

    void setInnerRegionCenter(const std::vector<size_t> &);
    void setForcedCoreList(const std::vector<int> &);
    void setForcedLayerList(const std::vector<int> &);
    void setForcedOuterList(const std::vector<int> &);

    void setUseQMCharges(bool useQMCharges);

    void setCoreRadius(double radius);
    void setLayerRadius(double radius);
    void setSmoothingRegionThickness(double thickness);
    void setPointChargeThickness(double radius);

    void setSmoothingMethod(SmoothingMethod method);
    void setQMForceDist(QMForceDist method);

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

#endif   // _HYBRID_SETTINGS_HPP_
