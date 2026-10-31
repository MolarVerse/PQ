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

#ifndef _MANOSTAT_SETTINGS_HPP_

#define _MANOSTAT_SETTINGS_HPP_

#include "enums/manostat.hpp"
#include "setting.hpp"
#include "unit_types.hpp"

/**
 * @class ManostatSettings
 *
 * @brief  class to store settings of the manosta constt
 *
 */
class ManostatSettings
{
   private:
    Setting<ManostatType> _manostatType;
    Setting<Isotropy>     _isotropy;
    Setting<FixedAxis>    _fixedAxis;

    Setting<double> _targetPressure;

    Setting<Time>   _tau;
    Setting<double> _compressibility;

   public:
    ManostatSettings();

    void finalize();

    /***************************
     * standard setter methods *
     ***************************/

    void setManostatType(ManostatType manostatType);
    void setIsotropy(Isotropy isotropy);

    void setFixedAxis(FixedAxis fixedAxis);

    void setTargetPressure(double targetPressure);
    void setTau(const Time& tau);
    void setCompressibility(double compressibility);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] bool isBerendsenBased() const;

    [[nodiscard]] ManostatType getManostatType() const;
    [[nodiscard]] Isotropy     getIsotropy() const;
    [[nodiscard]] FixedAxis    getFixedAxis() const;
    [[nodiscard]] bool         isFixedAxisSet() const;
    [[nodiscard]] double       getTargetPressure() const;
    [[nodiscard]] const Time&  getTau() const;
    [[nodiscard]] double       getCompressibility() const;
};

#endif   // _MANOSTAT_SETTINGS_HPP_
