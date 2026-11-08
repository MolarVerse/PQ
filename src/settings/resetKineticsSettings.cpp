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

#include "resetKineticsSettings.hpp"

#include "timingsSettings.hpp"

/**
 * @brief Construct a new Reset Kinetics Settings object
 *
 */
ResetKineticsSettings::ResetKineticsSettings()
    : _nScale(0),
      _fScale(0),
      _nReset(0),
      _fReset(0),
      _nResetAngular(0),
      _fResetAngular(0),
      _fResetForces(0)
{
}

/**
 * @brief Finalize the reset kinetics settings by setting default values for
 * certain fields if they are zero
 */
void ResetKineticsSettings::finalize()
{
    const auto numberOfSteps = settings::TimingsSettings::getNumberOfSteps();

    if (_fScale.get() == 0)
        _fScale.setBySystem(numberOfSteps + 1);

    if (_fReset.get() == 0)
        _fReset.setBySystem(numberOfSteps + 1);

    if (_fResetAngular.get() == 0)
        _fResetAngular.setBySystem(numberOfSteps + 1);

    if (_fResetForces.get() == 0)
        _fResetForces.setBySystem(numberOfSteps + 1);
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set nScale
 *
 * @param nScale
 */
void ResetKineticsSettings::setNScaleByUser(size_t nScale)
{
    _nScale.setByUser(nScale);
}

/**
 * @brief set fScale
 *
 * @param fScale
 */
void ResetKineticsSettings::setFScaleByUser(size_t fScale)
{
    _fScale.setByUser(fScale);
}

/**
 * @brief set nReset
 *
 * @param nReset
 */
void ResetKineticsSettings::setNResetByUser(size_t nReset)
{
    _nReset.setByUser(nReset);
}

/**
 * @brief set fReset
 *
 * @param fReset
 */
void ResetKineticsSettings::setFResetByUser(size_t fReset)
{
    _fReset.setByUser(fReset);
}

/**
 * @brief set nResetAngular
 *
 * @param nResetAngular
 */
void ResetKineticsSettings::setNResetAngularByUser(size_t nResetAngular)
{
    _nResetAngular.setByUser(nResetAngular);
}

/**
 * @brief set fResetAngular
 *
 * @param fResetAngular
 */
void ResetKineticsSettings::setFResetAngularByUser(size_t fResetAngular)
{
    _fResetAngular.setByUser(fResetAngular);
}

/**
 * @brief set fResetForces
 *
 * @param fResetForces
 */
void ResetKineticsSettings::setFResetForcesByUser(size_t fResetForces)
{
    _fResetForces.setByUser(fResetForces);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get nScale
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getNScale() const { return _nScale.get(); }

/**
 * @brief get fScale
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFScale() const { return _fScale.get(); }

/**
 * @brief get nReset
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getNReset() const { return _nReset.get(); }

/**
 * @brief get fReset
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFReset() const { return _fReset.get(); }

/**
 * @brief get nResetAngular
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getNResetAngular() const
{
    return _nResetAngular.get();
}

/**
 * @brief get fResetAngular
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFResetAngular() const
{
    return _fResetAngular.get();
}

/**
 * @brief get fResetForces
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFResetForces() const
{
    return _fResetForces.get();
}
