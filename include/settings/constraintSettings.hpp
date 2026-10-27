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

#include <cstddef>
#include <mstd/enum.hpp>

#include "setting.hpp"

/**
 * @class ConstraintSettings
 *
 * @brief static class to store settings of the constraints
 *
 */
class ConstraintSettings
{
   private:
    Setting<bool> _shakeActive;
    Setting<bool> _mShakeActive;
    Setting<bool> _distanceConstsActive;

    Setting<size_t> _shakeMaxIter;
    Setting<size_t> _rattleMaxIter;
    Setting<size_t> _mshakeMaxIter;

    Setting<double> _shakeTolerance;
    Setting<double> _rattleTolerance;
    Setting<double> _mshakeTolerance;

   public:
    ConstraintSettings();

    [[nodiscard]]
    bool isActive() const;

    [[nodiscard]]
    bool isShakeLikeActive() const;

    /*****************************
     * standard activate methods *
     *****************************/

    void activateShake();
    void deactivateShake();
    void activateMShake();
    void deactivateMShake();
    void activateDistanceConstraints();
    void deactivateDistanceConstraints();

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] bool isShakeActivated() const;
    [[nodiscard]] bool isMShakeActivated() const;
    [[nodiscard]] bool isDistanceConstraintsActivated() const;

    [[nodiscard]] size_t getShakeMaxIter() const;
    [[nodiscard]] size_t getRattleMaxIter() const;
    [[nodiscard]] size_t getMShakeMaxIter() const;
    [[nodiscard]] double getShakeTolerance() const;
    [[nodiscard]] double getRattleTolerance() const;
    [[nodiscard]] double getMShakeTolerance() const;

    /***************************
     * standard setter methods *
     ***************************/

    void setShakeMaxIter(size_t shakeMaxIter);
    void setRattleMaxIter(size_t rattleMaxIter);
    void setMShakeMaxIter(size_t mshakeMaxIter);
    void setShakeTolerance(double shakeTolerance);
    void setRattleTolerance(double rattleTolerance);
    void setMShakeTolerance(double mshakeTolerance);
};

#endif   // _CONSTRAINT_SETTINGS_HPP_
