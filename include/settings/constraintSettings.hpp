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

#ifndef _CONSTRAINT_SETTINGS_HPP_

#define _CONSTRAINT_SETTINGS_HPP_

#include <cstddef>   // for size_t

#include "defaults.hpp"

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
        static inline bool _shakeActivated =
            defaults::_CONSTRAINTS_ACTIVE_DEFAULT_;   // true
        static inline bool _distanceConstraintsActivated =
            defaults::_CONSTRAINTS_ACTIVE_DEFAULT_;   // true

        static inline size_t _shakeMaxIter =
            defaults::_SHAKE_MAX_ITER_DEFAULT_;   // 20
        static inline size_t _rattleMaxIter =
            defaults::_RATTLE_MAX_ITER_DEFAULT_;   // 20

        static inline double _shakeTolerance =
            defaults::_SHAKE_TOLERANCE_DEFAULT_;   // 1e-8
        static inline double _rattleTolerance =
            defaults::_RATTLE_TOLERANCE_DEFAULT_;   // 1e-8

       public:
        ConstraintSettings()  = default;
        ~ConstraintSettings() = default;

        static void activateShake() { _shakeActivated = true; }
        static void deactivateShake() { _shakeActivated = false; }
        static void activateDistanceConstraints()
        {
            _distanceConstraintsActivated = true;
        }
        static void deactivateDistanceConstraints()
        {
            _distanceConstraintsActivated = false;
        }

        static void setShakeMaxIter(const size_t shakeMaxIter)
        {
            _shakeMaxIter = shakeMaxIter;
        }
        static void setRattleMaxIter(const size_t rattleMaxIter)
        {
            _rattleMaxIter = rattleMaxIter;
        }
        static void setShakeTolerance(const double shakeTolerance)
        {
            _shakeTolerance = shakeTolerance;
        }
        static void setRattleTolerance(const double rattleTolerance)
        {
            _rattleTolerance = rattleTolerance;
        }

        [[nodiscard]] static bool isShakeActivated() { return _shakeActivated; }
        [[nodiscard]] static bool isDistanceConstraintsActivated()
        {
            return _distanceConstraintsActivated;
        }

        [[nodiscard]] static size_t getShakeMaxIter() { return _shakeMaxIter; }
        [[nodiscard]] static size_t getRattleMaxIter()
        {
            return _rattleMaxIter;
        }
        [[nodiscard]] static double getShakeTolerance()
        {
            return _shakeTolerance;
        }
        [[nodiscard]] static double getRattleTolerance()
        {
            return _rattleTolerance;
        }
    };

}   // namespace settings

#endif   // _CONSTRAINT_SETTINGS_HPP_