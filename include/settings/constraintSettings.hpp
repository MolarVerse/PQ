#ifndef _CONSTRAINT_SETTINGS_HPP_

#define _CONSTRAINT_SETTINGS_HPP_

#include "defaults.hpp"

#include <cstddef>   // for size_t

namespace settings
{
    /**
     * @class ConstraintSettings
     *
     * @brief static class to store settings of the constraints
     *
     */
    class ConstraintSettings
    {
      private:
        static inline size_t _shakeMaxIter  = defaults::_SHAKE_MAX_ITER_DEFAULT_;    // 20
        static inline size_t _rattleMaxIter = defaults::_RATTLE_MAX_ITER_DEFAULT_;   // 20

        static inline double _shakeTolerance  = defaults::_SHAKE_TOLERANCE_DEFAULT_;    // 1e-8
        static inline double _rattleTolerance = defaults::_RATTLE_TOLERANCE_DEFAULT_;   // 1e-8

      public:
        ConstraintSettings()  = default;
        ~ConstraintSettings() = default;

        static void setShakeMaxIter(const size_t shakeMaxIter) { _shakeMaxIter = shakeMaxIter; }
        static void setRattleMaxIter(const size_t rattleMaxIter) { _rattleMaxIter = rattleMaxIter; }
        static void setShakeTolerance(const double shakeTolerance) { _shakeTolerance = shakeTolerance; }
        static void setRattleTolerance(const double rattleTolerance) { _rattleTolerance = rattleTolerance; }

        [[nodiscard]] static size_t getShakeMaxIter() { return _shakeMaxIter; }
        [[nodiscard]] static size_t getRattleMaxIter() { return _rattleMaxIter; }
        [[nodiscard]] static double getShakeTolerance() { return _shakeTolerance; }
        [[nodiscard]] static double getRattleTolerance() { return _rattleTolerance; }
    };

}   // namespace settings

#endif   // _CONSTRAINT_SETTINGS_HPP_