#ifndef _MANOSTAT_SETTINGS_HPP_

#define _MANOSTAT_SETTINGS_HPP_

#include "defaults.hpp"

#include <string>        // for string
#include <string_view>   // for string_view

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
        static inline std::string _manostatType = defaults::_MANOSTAT_DEFAULT_;   // none

        static inline bool _isPressureSet = false;

        static inline double _targetPressure;   // no default value - has to be set by user
        static inline double _tauManostat     = defaults::_BERENDSEN_MANOSTAT_RELAXATION_TIME_;   // 1.0 ps
        static inline double _compressibility = defaults::_COMPRESSIBILITY_WATER_DEFAULT_;        // 4.5e-5 1/bar

      public:
        static void setManostatType(const std::string_view &manostatType) { _manostatType = manostatType; }
        static void setPressureSet(const bool pressureSet) { _isPressureSet = pressureSet; }
        static void setTargetPressure(const double targetPressure) { _targetPressure = targetPressure; }
        static void setTauManostat(const double tauManostat) { _tauManostat = tauManostat; }
        static void setCompressibility(const double compressibility) { _compressibility = compressibility; }

        [[nodiscard]] static std::string getManostatType() { return _manostatType; }
        [[nodiscard]] static bool        isPressureSet() { return _isPressureSet; }
        [[nodiscard]] static double      getTargetPressure() { return _targetPressure; }
        [[nodiscard]] static double      getTauManostat() { return _tauManostat; }
        [[nodiscard]] static double      getCompressibility() { return _compressibility; }
    };

}   // namespace settings

#endif   // _MANOSTAT_SETTINGS_HPP_