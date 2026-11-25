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
#include <mstd/utils/members.hpp>
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

    Setting<std::vector<size_t>, "Inner Region Center"> _innerRegionCenter{
        std::vector<size_t>{0}
    };
    Setting<std::vector<int>, "Forced Core List"> _forcedCoreList{
        std::vector<int>{}
    };
    Setting<std::vector<int>, "Forced Layer List"> _forcedLayerList{
        std::vector<int>{}
    };
    Setting<std::vector<int>, "Forced Outer List"> _forcedOuterList{
        std::vector<int>{}
    };
    std::optional<std::string> _innerRegionCenterExpression;

    Setting<bool, "Use QM Charges"> _useQMCharges{true};

    Setting<Distance, "Core Radius">  _coreRadius{0.0 * mstd::units::angstrom};
    Setting<Distance, "Layer Radius"> _layerRadius{0.0 * mstd::units::angstrom};
    Setting<Distance, "Smoothing Region Thickness"> _smoothingRegionThickness{
        0.0 * mstd::units::angstrom
    };
    Setting<Distance, "Point Charge Thickness"> _pointChargeThickness{
        0.0 * mstd::units::angstrom
    };

    Setting<SmoothingMethod, "Smoothing Method"> _smoothing{
        SmoothingMethod::HOTSPOT
    };
    Setting<QMForceDist, "QM Force Distribution"> _qmForceDist{
        QMForceDist::EQUAL
    };

   public:
    void finalize(const FileSettings &fileSettings, JobType jobType);

    SETTING_GETTER_AND_SETTER_BY_USER(_innerRegionCenter);
    CONST_GETTER_SETTER(_innerRegionCenterExpression);

    SETTING_GETTER_AND_SETTER_BY_USER(_forcedCoreList);
    SETTING_GETTER_AND_SETTER_BY_USER(_forcedLayerList);
    SETTING_GETTER_AND_SETTER_BY_USER(_forcedOuterList);

    SETTING_GETTER_AND_SETTER_BY_USER(_useQMCharges);

    SETTING_GETTER_AND_SETTER_BY_USER(_coreRadius);
    SETTING_GETTER_AND_SETTER_BY_USER(_layerRadius);
    SETTING_GETTER_AND_SETTER_BY_USER(_smoothingRegionThickness);
    SETTING_GETTER_AND_SETTER_BY_USER(_pointChargeThickness);

    SETTING_GETTER_AND_SETTER_BY_USER(_smoothing);
    SETTING_GETTER_AND_SETTER_BY_USER(_qmForceDist);

    [[nodiscard]] bool isActive() const;
};

#endif   // SETTINGS__HYBRID_SETTINGS_HPP
