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

#include "constraintSettings.hpp"

using namespace settings;

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activate the shake algorithm
 *
 */
void ConstraintSettings::activateShake() { _shakeActive = true; }

/**
 * @brief deactivate the shake algorithm
 *
 */
void ConstraintSettings::deactivateShake() { _shakeActive = false; }

/**
 * @brief activate the M-shake algorithm
 *
 */
void ConstraintSettings::activateMShake() { _mShakeActive = true; }

/**
 * @brief deactivate the M-shake algorithm
 *
 */
void ConstraintSettings::deactivateMShake() { _mShakeActive = false; }

/**
 * @brief activate the distance constraints
 *
 */
void ConstraintSettings::activateDistanceConstraints()
{
    _distanceConstsActive = true;
}

/**
 * @brief deactivate the distance constraints
 *
 */
void ConstraintSettings::deactivateDistanceConstraints()
{
    _distanceConstsActive = false;
}

/*****************************
 *                           *
 * standard getter methods   *
 *                           *
 *****************************/

/**
 * @brief check if the shake algorithm is activated
 *
 * @return true if shake is activated
 */
bool ConstraintSettings::isShakeActivated() { return _shakeActive; }

/**
 * @brief check if the M-shake algorithm is activated
 *
 * @return true if M-shake is activated
 */
bool ConstraintSettings::isMShakeActivated() { return _mShakeActive; }

/**
 * @brief check if the distance constraints are activated
 *
 * @return true if distance constraints are activated
 */
bool ConstraintSettings::isDistanceConstraintsActivated()
{
    return _distanceConstsActive;
}

/**
 * @brief get the maximum number of iterations for the shake algorithm
 *
 * @return the maximum number of iterations
 */
size_t ConstraintSettings::getShakeMaxIter() { return _shakeMaxIter; }

/**
 * @brief get the maximum number of iterations for the rattle algorithm
 *
 * @return the maximum number of iterations
 */
size_t ConstraintSettings::getRattleMaxIter() { return _rattleMaxIter; }

/**
 * @brief get the tolerance for the shake algorithm
 *
 * @return the tolerance
 */
double ConstraintSettings::getShakeTolerance() { return _shakeTolerance; }

/**
 * @brief get the tolerance for the rattle algorithm
 *
 * @return the tolerance
 */
double ConstraintSettings::getRattleTolerance() { return _rattleTolerance; }

/*****************************
 *                           *
 * standard setter methods   *
 *                           *
 *****************************/

/**
 * @brief set the maximum number of iterations for the shake algorithm
 *
 * @param shakeMaxIter
 */
void ConstraintSettings::setShakeMaxIter(const size_t shakeMaxIter)
{
    _shakeMaxIter = shakeMaxIter;
}

/**
 * @brief set the maximum number of iterations for the rattle algorithm
 *
 * @param rattleMaxIter
 */
void ConstraintSettings::setRattleMaxIter(const size_t rattleMaxIter)
{
    _rattleMaxIter = rattleMaxIter;
}

/**
 * @brief set the tolerance for the shake algorithm
 *
 * @param shakeTolerance
 */
void ConstraintSettings::setShakeTolerance(const double shakeTolerance)
{
    _shakeTolerance = shakeTolerance;
}

/**
 * @brief set the tolerance for the rattle algorithm
 *
 * @param rattleTolerance
 */
void ConstraintSettings::setRattleTolerance(const double rattleTolerance)
{
    _rattleTolerance = rattleTolerance;
}