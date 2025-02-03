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

using settings::ResetKineticsSettings;

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
void ResetKineticsSettings::setNScale(const size_t nScale) { _nScale = nScale; }

/**
 * @brief set fScale
 *
 * @param fScale
 */
void ResetKineticsSettings::setFScale(const size_t fScale) { _fScale = fScale; }

/**
 * @brief set nReset
 *
 * @param nReset
 */
void ResetKineticsSettings::setNReset(const size_t nReset) { _nReset = nReset; }

/**
 * @brief set fReset
 *
 * @param fReset
 */
void ResetKineticsSettings::setFReset(const size_t fReset) { _fReset = fReset; }

/**
 * @brief set nResetAngular
 *
 * @param nResetAngular
 */
void ResetKineticsSettings::setNResetAngular(const size_t nResetAngular)
{
    _nResetAngular = nResetAngular;
}

/**
 * @brief set fResetAngular
 *
 * @param fResetAngular
 */
void ResetKineticsSettings::setFResetAngular(const size_t fResetAngular)
{
    _fResetAngular = fResetAngular;
}

/**
 * @brief set fResetForces
 *
 * @param fResetForces
 */
void ResetKineticsSettings::setFResetForces(const size_t fResetForces)
{
    _fResetForces = fResetForces;
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
size_t ResetKineticsSettings::getNScale() { return _nScale; }

/**
 * @brief get fScale
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFScale() { return _fScale; }

/**
 * @brief get nReset
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getNReset() { return _nReset; }

/**
 * @brief get fReset
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFReset() { return _fReset; }

/**
 * @brief get nResetAngular
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getNResetAngular() { return _nResetAngular; }

/**
 * @brief get fResetAngular
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFResetAngular() { return _fResetAngular; }

/**
 * @brief get fResetForces
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getFResetForces() { return _fResetForces; }