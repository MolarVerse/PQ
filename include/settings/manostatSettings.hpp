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

#include <string_view>   // for string_view
#include <vector>        // for vector

#include "defaults.hpp"
#include "enums/manostat.hpp"

namespace settings
{
    /**
     * @class ManostatSettings
     *
     * @brief static class to store settings of the manostat
     *
     */
    class ManostatSettings
    {
       private:
        static inline ManostatType _manostatType   = ManostatType::NONE;
        static inline Isotropy     _isotropy       = Isotropy::ISOTROPIC;
        static inline FixedAxis    _fixedAxis      = FixedAxis::ALL;
        static inline bool         _isFixedAxisSet = false;

        static inline double _targetPressure;

        static inline double _tauManostat =
            defaults::BERENDSEN_MANOSTAT_RELAX_TIME;
        static inline double _compressibility =
            defaults::COMPRESSIBILITY_WATER_DEFAULT;

       public:
        ManostatSettings()  = default;
        ~ManostatSettings() = default;

        /***************************
         * standard setter methods *
         ***************************/

        static void setManostatType(ManostatType manostatType);
        static void setIsotropy(Isotropy isotropy);

        static void setFixedAxis(FixedAxis fixedAxis);
        static void setIsFixedAxisSet(bool isSet);

        static void setTargetPressure(double targetPressure);
        static void setTauManostat(double tauManostat);
        static void setCompressibility(double compressibility);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static bool isBerendsenBased();

        [[nodiscard]] static ManostatType getManostatType();
        [[nodiscard]] static Isotropy     getIsotropy();
        [[nodiscard]] static FixedAxis    getFixedAxis();
        [[nodiscard]] static bool         isFixedAxisSet();
        [[nodiscard]] static double       getTargetPressure();
        [[nodiscard]] static double       getTauManostat();
        [[nodiscard]] static double       getCompressibility();
    };

}   // namespace settings

#endif   // _MANOSTAT_SETTINGS_HPP_
