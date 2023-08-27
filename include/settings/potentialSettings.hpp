#ifndef _POSITION_SETTINGS_HPP_

#define _POSITION_SETTINGS_HPP_

#include "defaults.hpp"

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
        static inline double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;
        static inline double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;

      public:
        static void setScale14Coulomb(double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
        static void setScale14VanDerWaals(double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }

        [[nodiscard]] static double getScale14Coulomb() { return _scale14Coulomb; }
        [[nodiscard]] static double getScale14VanDerWaals() { return _scale14VanDerWaals; }
    };

}   // namespace settings

#endif   // _POSITION_SETTINGS_HPP_