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

#include "defaults.hpp"

/**
 * @brief Construct a new Constraint Settings object
 *
 * Initializes the constraint settings with default values.
 */
ConstraintSettings::ConstraintSettings()
    : _shakeActive(defaults::CONSTRAINTS_ACTIVE_DEFAULT),
      _mShakeActive(defaults::CONSTRAINTS_ACTIVE_DEFAULT),
      _distanceConstsActive(defaults::CONSTRAINTS_ACTIVE_DEFAULT),
      _shakeMaxIter(defaults::SHAKE_MAX_ITER_DEFAULT),
      _rattleMaxIter(defaults::RATTLE_MAX_ITER_DEFAULT),
      _mshakeMaxIter(defaults::MSHAKE_MAX_ITER_DEFAULT),
      _shakeTolerance(defaults::SHAKE_TOLERANCE_DEFAULT),
      _rattleTolerance(defaults::RATTLE_TOLERANCE_DEFAULT),
      _mshakeTolerance(defaults::MSHAKE_TOLERANCE_DEFAULT)
{
}

/**
 * @brief check if the shake algorithm is only active for the water model
 *
 * @return true if shake is only active for the water model
 */
bool ConstraintSettings::isShakeOnlyWaterModel() const
{
    return _shakeOnlyWaterModel;
}

/**
 * @brief activate the shake algorithm only for the water model
 *
 */
void ConstraintSettings::activateShakeForWaterModel()
{
    if (!_shakeActive.get())
        _shakeOnlyWaterModel = true;

    _shakeActive.set(true);
}

/**
 * @brief check if any of the constraints algorithms are activated
 *
 * @return true if any of the constraints algorithms are activated
 */
bool ConstraintSettings::isActive() const
{
    return _shakeActive.get() || _mShakeActive.get() ||
           _distanceConstsActive.get();
}

/**
 * @brief check if any of the shake-like algorithms (shake or mShake) are
 * activated
 *
 * @return true if any of the shake-like algorithms are activated
 */
bool ConstraintSettings::isShakeLikeActive() const
{
    return _shakeActive.get() || _mShakeActive.get();
}

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activate the shake algorithm
 *
 */
void ConstraintSettings::activateShake() { _shakeActive.set(true); }

/**
 * @brief deactivate the shake algorithm
 *
 */
void ConstraintSettings::deactivateShake() { _shakeActive.set(false); }

/**
 * @brief activate the M-shake algorithm
 *
 */
void ConstraintSettings::activateMShake() { _mShakeActive.set(true); }

/**
 * @brief deactivate the M-shake algorithm
 *
 */
void ConstraintSettings::deactivateMShake() { _mShakeActive.set(false); }

/**
 * @brief activate the distance constraints
 *
 */
void ConstraintSettings::activateDistanceConstraints()
{
    _distanceConstsActive.set(true);
}

/**
 * @brief deactivate the distance constraints
 *
 */
void ConstraintSettings::deactivateDistanceConstraints()
{
    _distanceConstsActive.set(false);
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
bool ConstraintSettings::isShakeActivated() const { return _shakeActive.get(); }

/**
 * @brief check if the M-shake algorithm is activated
 *
 * @return true if M-shake is activated
 */
bool ConstraintSettings::isMShakeActivated() const
{
    return _mShakeActive.get();
}

/**
 * @brief check if the distance constraints are activated
 *
 * @return true if distance constraints are activated
 */
bool ConstraintSettings::isDistanceConstraintsActivated() const
{
    return _distanceConstsActive.get();
}

/**
 * @brief get the maximum number of iterations for the shake algorithm
 *
 * @return the maximum number of iterations
 */
size_t ConstraintSettings::getShakeMaxIter() const
{
    return _shakeMaxIter.get();
}

/**
 * @brief get the maximum number of iterations for the rattle algorithm
 *
 * @return the maximum number of iterations
 */
size_t ConstraintSettings::getRattleMaxIter() const
{
    return _rattleMaxIter.get();
}

/**
 * @brief get the maximum number of iterations for the MShake algorithm
 *
 * @return the maximum number of iterations
 */
size_t ConstraintSettings::getMShakeMaxIter() const
{
    return _mshakeMaxIter.get();
}

/**
 * @brief get the tolerance for the shake algorithm
 *
 * @return the tolerance
 */
double ConstraintSettings::getShakeTolerance() const
{
    return _shakeTolerance.get();
}

/**
 * @brief get the tolerance for the rattle algorithm
 *
 * @return the tolerance
 */
double ConstraintSettings::getRattleTolerance() const
{
    return _rattleTolerance.get();
}

/**
 * @brief get the tolerance for the MShake algorithm
 *
 * @return the tolerance
 */
double ConstraintSettings::getMShakeTolerance() const
{
    return _mshakeTolerance.get();
}

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
void ConstraintSettings::setShakeMaxIter(size_t shakeMaxIter)
{
    _shakeMaxIter.set(shakeMaxIter);
}

/**
 * @brief set the maximum number of iterations for the rattle algorithm
 *
 * @param rattleMaxIter
 */
void ConstraintSettings::setRattleMaxIter(size_t rattleMaxIter)
{
    _rattleMaxIter.set(rattleMaxIter);
}

/**
 * @brief set the maximum number of iterations for the MShake algorithm
 *
 * @param mshakeMaxIter
 */
void ConstraintSettings::setMShakeMaxIter(size_t mshakeMaxIter)
{
    _mshakeMaxIter.set(mshakeMaxIter);
}

/**
 * @brief set the tolerance for the shake algorithm
 *
 * @param shakeTolerance
 */
void ConstraintSettings::setShakeTolerance(double shakeTolerance)
{
    _shakeTolerance.set(shakeTolerance);
}

/**
 * @brief set the tolerance for the rattle algorithm
 *
 * @param rattleTolerance
 */
void ConstraintSettings::setRattleTolerance(double rattleTolerance)
{
    _rattleTolerance.set(rattleTolerance);
}

/**
 * @brief set the tolerance for the MShake algorithm
 *
 * @param mshakeTolerance
 */
void ConstraintSettings::setMShakeTolerance(double mshakeTolerance)
{
    _mshakeTolerance.set(mshakeTolerance);
}
