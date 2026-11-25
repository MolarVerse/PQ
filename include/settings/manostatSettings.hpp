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

    /***************************
     * standard setter methods *
     ***************************/

    void setManostatTypeByUser(ManostatType manostatType);
    void setIsotropyByUser(Isotropy isotropy);

    void setFixedAxisByUser(FixedAxis fixedAxis);

    void setTargetPressureByUser(const Pressure& targetPressure);
    void setTauByUser(const Time& tau);
    void setCompressibilityByUser(const Compressibility& compressibility);

    void setVirialTypeByUser(VirialType virialType);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] bool isBerendsenBased() const;

    [[nodiscard]] ManostatType           getManostatType() const;
    [[nodiscard]] Isotropy               getIsotropy() const;
    [[nodiscard]] FixedAxis              getFixedAxis() const;
    [[nodiscard]] const Pressure&        getTargetPressure() const;
    [[nodiscard]] const Time&            getTau() const;
    [[nodiscard]] const Compressibility& getCompressibility() const;

    [[nodiscard]] VirialType getVirialType() const;

   private:
    void _addReferenceFile(ReferenceSettings& referenceSettings);
};

#endif   // SETTINGS__MANOSTAT_SETTINGS_HPP
