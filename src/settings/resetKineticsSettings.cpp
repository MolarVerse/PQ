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

#include <algorithm>
#include <cstddef>
#include <optional>

#include "defaults.hpp"

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
      _fResetForces(0),
      _momentumWarnThreshold(defaults::MOMENTUM_WARN_THRESHOLD_DEFAULT)
{
}

/**
 * @brief Finalize the reset kinetics settings by setting default values for
 * certain fields if they are zero
 */
void ResetKineticsSettings::finalize(size_t numberOfSteps)
{
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

/**
 * @brief set the momentum drift warning threshold
 *
 * @param threshold fraction of the thermal velocity, 0 disables the warning
 */
void ResetKineticsSettings::setMomentumWarnThresholdByUser(double threshold)
{
    _momentumWarnThreshold.setByUser(threshold);
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

/**
 * @brief get the momentum drift warning threshold
 *
 * @return double
 */
double ResetKineticsSettings::getMomentumWarnThreshold() const
{
    return _momentumWarnThreshold.get();
}

/**
 * @brief number of first steps in which nReset or nScale removes the momentum
 *
 * @return size_t
 */
size_t ResetKineticsSettings::getInitialMomentumResetSteps() const
{
    return std::max(_nReset.get(), _nScale.get());
}

/**
 * @brief smallest interval of the periodic momentum reset by fReset or fScale
 *
 * @return the interval, std::nullopt if neither is set by the user
 */
std::optional<size_t> ResetKineticsSettings::getMomentumResetInterval() const
{
    std::optional<size_t> interval;

    for (const auto *setting : {&_fReset, &_fScale})
    {
        if (setting->isSetByUser() && setting->get() > 0)
        {
            interval =
                std::min(interval.value_or(setting->get()), setting->get());
        }
    }

    return interval;
}
