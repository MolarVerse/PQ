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

#include <mstd/units/new_units.hpp>
#include <optional>
#include <string>
#include <vector>

#include "enums/hybrid.hpp"
#include "enums/jobtype.hpp"
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
    bool _isActive{false};

    Setting<std::vector<size_t>> _innerRegionCenter{std::vector<size_t>{0}};
    Setting<std::vector<int>>    _forcedCoreList{std::vector<int>{}};
    Setting<std::vector<int>>    _forcedLayerList{std::vector<int>{}};
    Setting<std::vector<int>>    _forcedOuterList{std::vector<int>{}};
    std::optional<std::string>   _innerRegionCenterExpression;

    Setting<bool> _useQMCharges{true};

    Setting<Distance> _coreRadius{0.0 * mstd::units::angstrom};
    Setting<Distance> _layerRadius{0.0 * mstd::units::angstrom};
    Setting<Distance> _smoothingRegionThickness{0.0 * mstd::units::angstrom};
    Setting<Distance> _pointChargeThickness{0.0 * mstd::units::angstrom};

    Setting<SmoothingMethod> _smoothing{SmoothingMethod::HOTSPOT};
    Setting<QMForceDist>     _qmForceDist{QMForceDist::EQUAL};

   public:
    void finalize(const FileSettings &fileSettings, JobType jobType);

    /********************
     * standard setters *
     ********************/

    void setInnerRegionCenterByUser(const std::vector<size_t> &);
    void setInnerRegionCenterExpression(const std::string &);
    void setForcedCoreListByUser(const std::vector<int> &);
    void setForcedLayerListByUser(const std::vector<int> &);
    void setForcedOuterListByUser(const std::vector<int> &);

    void setUseQMChargesByUser(bool useQMCharges);

    void setCoreRadiusByUser(const Distance &radius);
    void setLayerRadiusByUser(const Distance &radius);
    void setSmoothingRegionThicknessByUser(const Distance &thickness);
    void setPointChargeThicknessByUser(const Distance &radius);

    void setSmoothingMethodByUser(SmoothingMethod method);
    void setQMForceDistByUser(QMForceDist method);

    /********************
     * standard getters *
     ********************/

    [[nodiscard]] const std::vector<size_t> &getInnerRegionCenter() const;
    [[nodiscard]] const std::vector<int>    &getForcedCoreList() const;
    [[nodiscard]] const std::vector<int>    &getForcedLayerList() const;
    [[nodiscard]] const std::vector<int>    &getForcedOuterList() const;

    [[nodiscard]] bool getUseQMCharges() const;

    [[nodiscard]] const Distance &getCoreRadius() const;
    [[nodiscard]] const Distance &getLayerRadius() const;
    [[nodiscard]] const Distance &getSmoothingRegionThickness() const;
    [[nodiscard]] const Distance &getPointChargeThickness() const;

    [[nodiscard]] SmoothingMethod getSmoothingMethod() const;
    [[nodiscard]] QMForceDist     getQMForceDist() const;

    [[nodiscard]] bool isActive() const;
};

#endif   // SETTINGS__HYBRID_SETTINGS_HPP
