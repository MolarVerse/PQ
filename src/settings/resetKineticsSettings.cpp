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
        _fScale.set(numberOfSteps + 1);

    if (_fReset.get() == 0)
        _fReset.set(numberOfSteps + 1);

    if (_fResetAngular.get() == 0)
        _fResetAngular.set(numberOfSteps + 1);

    if (_fResetForces.get() == 0)
        _fResetForces.set(numberOfSteps + 1);
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
void ResetKineticsSettings::setNScale(size_t nScale) { _nScale.set(nScale); }

/**
 * @brief set fScale
 *
 * @param fScale
 */
void ResetKineticsSettings::setFScale(size_t fScale) { _fScale.set(fScale); }

/**
 * @brief set nReset
 *
 * @param nReset
 */
void ResetKineticsSettings::setNReset(size_t nReset) { _nReset.set(nReset); }

/**
 * @brief set fReset
 *
 * @param fReset
 */
void ResetKineticsSettings::setFReset(size_t fReset) { _fReset.set(fReset); }

/**
 * @brief set nResetAngular
 *
 * @param nResetAngular
 */
void ResetKineticsSettings::setNResetAngular(size_t nResetAngular)
{
    _nResetAngular.set(nResetAngular);
}

/**
 * @brief set fResetAngular
 *
 * @param fResetAngular
 */
void ResetKineticsSettings::setFResetAngular(size_t fResetAngular)
{
    _fResetAngular.set(fResetAngular);
}

/**
 * @brief set fResetForces
 *
 * @param fResetForces
 */
void ResetKineticsSettings::setFResetForces(size_t fResetForces)
{
    _fResetForces.set(fResetForces);
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
