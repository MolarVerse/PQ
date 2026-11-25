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

#ifndef SETTINGS__MANOSTAT_SETTINGS_HPP
#define SETTINGS__MANOSTAT_SETTINGS_HPP

#include "enums/manostat.hpp"
#include "enums/virial.hpp"
#include "setting.hpp"
#include "unit_types.hpp"

class ReferenceSettings;

/**
 * @class ManostatSettings
 *
 * @brief  class to store settings of the manosta constt
 *
 */
class ManostatSettings
{
   private:
    Setting<ManostatType, "Manostat Type"> _manostatType;
    Setting<Isotropy, "Isotropy">          _isotropy;
    Setting<FixedAxis, "Fixed Axis">       _fixedAxis;

    Setting<Pressure, "Target Pressure"> _targetPressure;

    Setting<Time, "Tau">                        _tau;
    Setting<Compressibility, "Compressibility"> _compressibility;

    Setting<VirialType, "VirialType"> _virial{VirialType::MOLECULAR};

   public:
    ManostatSettings();

    void finalize(ReferenceSettings& referenceSettings);

    SETTING_GETTER_AND_SETTER_BY_USER(_manostatType);
    SETTING_GETTER_AND_SETTER_BY_USER(_isotropy);
    SETTING_GETTER_AND_SETTER_BY_USER(_fixedAxis);
    SETTING_GETTER_AND_SETTER_BY_USER(_targetPressure);
    SETTING_GETTER_AND_SETTER_BY_USER(_tau);
    SETTING_GETTER_AND_SETTER_BY_USER(_compressibility);
    SETTING_GETTER_AND_SETTER_BY_USER(_virial);

    [[nodiscard]] bool isBerendsenBased() const;

   private:
    void _addReferenceFile(ReferenceSettings& referenceSettings);
};

#endif   // SETTINGS__MANOSTAT_SETTINGS_HPP
