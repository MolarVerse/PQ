#ifndef _POSITION_SETTINGS_HPP_

#define _POSITION_SETTINGS_HPP_

#include "defaults.hpp"   // for _COULOMB_LONG_RANGE_TYPE_DEFAULT_, ...

#include <string>        // for allocator, string
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @class PotentialSettings
     *
     * @brief static class to store settings of the potential
     *
     */
    class PotentialSettings
    {
      private:
        static inline std::string _coulombLongRangeType = defaults::_COULOMB_LONG_RANGE_TYPE_DEFAULT_;   // guff
        static inline std::string _nonCoulombType       = defaults::_NON_COULOMB_TYPE_DEFAULT_;   // none = shifted potential

        static inline double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;         // default is 1.0
        static inline double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;   // default is 1.0

        static inline double _wolfParameter = defaults::_WOLF_PARAMETER_DEFAULT_;   // default is 0.25

      public:
        static void setCoulombLongRangeType(const std::string_view &type) { _coulombLongRangeType = type; }
        static void setNonCoulombType(const std::string_view &type) { _nonCoulombType = type; }

        static void setScale14Coulomb(double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
        static void setScale14VanDerWaals(double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }
        static void setWolfParameter(double wolfParameter) { _wolfParameter = wolfParameter; }

        [[nodiscard]] static std::string getCoulombLongRangeType() { return _coulombLongRangeType; }
        [[nodiscard]] static std::string getNonCoulombType() { return _nonCoulombType; }

        [[nodiscard]] static double getScale14Coulomb() { return _scale14Coulomb; }
        [[nodiscard]] static double getScale14VanDerWaals() { return _scale14VanDerWaals; }
        [[nodiscard]] static double getWolfParameter() { return _wolfParameter; }
    };

}   // namespace settings

#endif   // _POSITION_SETTINGS_HPP_