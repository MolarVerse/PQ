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

#ifndef SETTINGS__CONSTRAINT_SETTINGS_HPP
#define SETTINGS__CONSTRAINT_SETTINGS_HPP

#include <cstddef>
#include <mstd/enum.hpp>

#include "setting.hpp"
#include "unit_types.hpp"

class ReferenceSettings;
class WaterModelSettings;

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

    Setting<double>      _shakeTolerance;
    Setting<MassPerTime> _rattleTolerance;
    Setting<double>      _mshakeTolerance;

   public:
    ConstraintSettings();

    void finalize(
        ReferenceSettings&        referenceSettings,
        const WaterModelSettings& waterModelSettings
    );

    SETTING_GETTER_AND_SETTER_BY_USER(_shakeMaxIter);
    SETTING_GETTER_AND_SETTER_BY_USER(_rattleMaxIter);
    SETTING_GETTER_AND_SETTER_BY_USER(_mshakeMaxIter);
    SETTING_GETTER_AND_SETTER_BY_USER(_shakeTolerance);
    SETTING_GETTER_AND_SETTER_BY_USER(_rattleTolerance);
    SETTING_GETTER_AND_SETTER_BY_USER(_mshakeTolerance);

    void activateShakeForWaterModel();

    [[nodiscard]]
    bool isActive() const;

    [[nodiscard]]
    bool isShakeLikeActive() const;

    [[nodiscard]]
    bool isShakeSetBySystem() const;

    /*****************************
     * standard activate methods *
     *****************************/

    void activateShakeByUser();
    void deactivateShakeByUser();
    void activateMShakeByUser();
    void deactivateMShakeByUser();
    void activateDistanceConstraintsByUser();
    void deactivateDistanceConstraintsByUser();

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] bool isShakeActivated() const;
    [[nodiscard]] bool isMShakeActivated() const;
    [[nodiscard]] bool isDistanceConstraintsActivated() const;
};

#endif   // SETTINGS__CONSTRAINT_SETTINGS_HPP
