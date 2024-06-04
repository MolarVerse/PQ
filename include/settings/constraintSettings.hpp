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
    using namespace defaults;

    /**
     * @class ConstraintSettings
     *
     * @brief static class to store settings of the constraints
     *
     */
    class ConstraintSettings
    {
       private:
        static inline bool _shakeActive          = _CONSTRAINTS_ACTIVE_DEFAULT_;
        static inline bool _mShakeActive         = _CONSTRAINTS_ACTIVE_DEFAULT_;
        static inline bool _distanceConstsActive = _CONSTRAINTS_ACTIVE_DEFAULT_;

        static inline size_t _shakeMaxIter  = _SHAKE_MAX_ITER_DEFAULT_;
        static inline size_t _rattleMaxIter = _RATTLE_MAX_ITER_DEFAULT_;

        static inline double _shakeTolerance  = _SHAKE_TOLERANCE_DEFAULT_;
        static inline double _rattleTolerance = _RATTLE_TOLERANCE_DEFAULT_;

       public:
        ConstraintSettings()  = default;
        ~ConstraintSettings() = default;

        /*****************************
         * standard activate methods *
         *****************************/

        static void activateShake();
        static void deactivateShake();
        static void activateMShake();
        static void deactivateMShake();
        static void activateDistanceConstraints();
        static void deactivateDistanceConstraints();

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static bool isShakeActivated();
        [[nodiscard]] static bool isMShakeActivated();
        [[nodiscard]] static bool isDistanceConstraintsActivated();

        [[nodiscard]] static size_t getShakeMaxIter();
        [[nodiscard]] static size_t getRattleMaxIter();
        [[nodiscard]] static double getShakeTolerance();
        [[nodiscard]] static double getRattleTolerance();

        /***************************
         * standard setter methods *
         ***************************/

        static void setShakeMaxIter(const size_t shakeMaxIter);
        static void setRattleMaxIter(const size_t rattleMaxIter);
        static void setShakeTolerance(const double shakeTolerance);
        static void setRattleTolerance(const double rattleTolerance);
    };

}   // namespace settings

#endif   // _CONSTRAINT_SETTINGS_HPP_