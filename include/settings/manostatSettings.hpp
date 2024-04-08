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

#include "defaults.hpp"

#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

namespace settings
{
    /**
     * @enum ManostatType
     *
     * @brief enum class to store the type of the manostat
     *
     */
    enum class ManostatType
    {
        NONE,
        BERENDSEN,
        STOCHASTIC_RESCALING
    };

    /**
     * @enum Isotropy
     *
     * @brief enum class to store the isotropy of the manostat
     *
     */
    enum class Isotropy
    {
        ISOTROPIC,
        SEMI_ISOTROPIC,
        ANISOTROPIC,
        FULL_ANISOTROPIC
    };

    [[nodiscard]] std::string string(const ManostatType &manostatType);
    [[nodiscard]] std::string string(const Isotropy &isotropy);

    /**
     * @class ManostatSettings
     *
     * @brief static class to store settings of the manostat
     *
     */
    class ManostatSettings
    {
      private:
        static inline ManostatType _manostatType = ManostatType::NONE;
        static inline Isotropy     _isotropy     = Isotropy::ISOTROPIC;

        static inline bool _isPressureSet = false;

        static inline double _targetPressure;   // no default value - has to be set by user
        static inline double _tauManostat     = defaults::_BERENDSEN_MANOSTAT_RELAXATION_TIME_;   // 1.0 ps
        static inline double _compressibility = defaults::_COMPRESSIBILITY_WATER_DEFAULT_;        // 4.5e-5 1/bar

        static inline std::vector<size_t> _2DIsotropicAxes;
        static inline size_t              _2DAnisotropicAxis;

      public:
        ManostatSettings()  = default;
        ~ManostatSettings() = default;

        static void setManostatType(const std::string_view &manostatType);
        static void setManostatType(const ManostatType &manostatType) { _manostatType = manostatType; }

        static void setIsotropy(const std::string_view &isotropy);
        static void setIsotropy(const Isotropy &isotropy) { _isotropy = isotropy; }

        /***************************
         * standard setter methods *
         ***************************/

        static void setPressureSet(const bool pressureSet) { _isPressureSet = pressureSet; }
        static void setTargetPressure(const double targetPressure) { _targetPressure = targetPressure; }
        static void setTauManostat(const double tauManostat) { _tauManostat = tauManostat; }
        static void setCompressibility(const double compressibility) { _compressibility = compressibility; }
        static void set2DIsotropicAxes(const std::vector<size_t> &indices) { _2DIsotropicAxes = indices; }
        static void set2DAnisotropicAxis(const size_t index) { _2DAnisotropicAxis = index; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static ManostatType getManostatType() { return _manostatType; }
        [[nodiscard]] static Isotropy     getIsotropy() { return _isotropy; }
        [[nodiscard]] static bool         isPressureSet() { return _isPressureSet; }
        [[nodiscard]] static double       getTargetPressure() { return _targetPressure; }
        [[nodiscard]] static double       getTauManostat() { return _tauManostat; }
        [[nodiscard]] static double       getCompressibility() { return _compressibility; }

        [[nodiscard]] static std::vector<size_t> get2DIsotropicAxes() { return _2DIsotropicAxes; }
        [[nodiscard]] static size_t              get2DAnisotropicAxis() { return _2DAnisotropicAxis; }
    };

}   // namespace settings

#endif   // _MANOSTAT_SETTINGS_HPP_