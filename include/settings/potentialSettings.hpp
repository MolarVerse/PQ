#ifndef _POSITION_SETTINGS_HPP_

#define _POSITION_SETTINGS_HPP_

#include "defaults.hpp"   // for _COULOMB_LONG_RANGE_TYPE_DEFAULT_, ...

#include <cstddef>       // for size_t
#include <string>        // for allocator, string
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @enum CoulombLongRangeType
     *
     * @brief enum class to store the coulomb long range type
     *
     */
    enum class NonCoulombType : size_t
    {
        LJ,
        LJ_9_12,   // at the momentum just dummy for testing not implemented yet
        BUCKINGHAM,
        MORSE,
        GUFF,
        NONE
    };

    std::string string(const NonCoulombType nonCoulombType);

    /**
     * @class PotentialSettings
     *
     * @brief static class to store settings of the potential
     *
     */
    class PotentialSettings
    {
      private:
        static inline std::string    _coulombLongRangeType = defaults::_COULOMB_LONG_RANGE_TYPE_DEFAULT_;   // shifted potential
        static inline std::string    _nonCoulombTypeString = defaults::_NON_COULOMB_TYPE_DEFAULT_;          // guff
        static inline NonCoulombType _nonCoulombType       = NonCoulombType::GUFF;                          // LJ

        static inline double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;         // default is 1.0
        static inline double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;   // default is 1.0

        static inline double _wolfParameter = defaults::_WOLF_PARAMETER_DEFAULT_;   // default is 0.25

      public:
        static void setNonCoulombType(const std::string_view &type);
        static void setNonCoulombType(const NonCoulombType type) { _nonCoulombType = type; }

        /********************
         * standard setters *
         ********************/

        static void setCoulombLongRangeType(const std::string_view &type) { _coulombLongRangeType = type; }

        static void setScale14Coulomb(double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
        static void setScale14VanDerWaals(double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }
        static void setWolfParameter(double wolfParameter) { _wolfParameter = wolfParameter; }

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static std::string    getCoulombLongRangeType() { return _coulombLongRangeType; }
        [[nodiscard]] static std::string    getNonCoulombTypeString() { return _nonCoulombTypeString; }
        [[nodiscard]] static NonCoulombType getNonCoulombType() { return _nonCoulombType; }

        [[nodiscard]] static double getScale14Coulomb() { return _scale14Coulomb; }
        [[nodiscard]] static double getScale14VanDerWaals() { return _scale14VanDerWaals; }
        [[nodiscard]] static double getWolfParameter() { return _wolfParameter; }
    };

}   // namespace settings

#endif   // _POSITION_SETTINGS_HPP_