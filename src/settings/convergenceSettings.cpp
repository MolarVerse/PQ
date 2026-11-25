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

#include "convergenceSettings.hpp"

#include "defaults.hpp"

/**
 * @brief Construct a new Conv Settings object
 *
 * @details Initializes the convergence settings with default values.
 */
ConvSettings::ConvSettings()
    : _useEnergyConv(true),
      _useForceConv(true),
      _useMaxForceConv(true),
      _useRMSForceConv(true),
      _energyConvStrategy(defaults::EN_CONV_STRATEGY_DEFAULT)
{
}

/**
 * @brief check if any convergence threshold was explicitly set by the user
 *
 * @return bool
 */
bool ConvSettings::_anyConvergenceSetByUser() const
{
    return _relEnergyConv.isSetByUser() || _absEnergyConv.isSetByUser() ||
           _maxForceConv.isSetByUser() || _rmsForceConv.isSetByUser();
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get use energy convergence
 *
 * @return bool
 */
bool ConvSettings::getUseEnergyConv() const
{
    if (_useEnergyConv.isSetBySomeone())
        return _useEnergyConv.get();

    if (_relEnergyConv.isSetByUser() || _absEnergyConv.isSetByUser())
        return true;

    // some other criterion was requested explicitly -> only use those
    if (_anyConvergenceSetByUser())
        return false;

    return _useEnergyConv.get();   // nothing specified -> default
}

/**
 * @brief get use force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseForceConv() const
{
    if (_useForceConv.isSetBySomeone())
        return _useForceConv.get();

    if (_maxForceConv.isSetByUser() || _rmsForceConv.isSetByUser())
        return true;

    // some other criterion was requested explicitly -> only use those
    if (_anyConvergenceSetByUser())
        return false;

    return _useForceConv.get();   // nothing specified -> default
}

/**
 * @brief get use max force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseMaxForceConv() const
{
    if (_useMaxForceConv.isSetBySomeone())
        return _useMaxForceConv.get();

    if (_maxForceConv.isSetByUser())
        return true;

    // some other criterion was requested explicitly -> only use those
    if (_anyConvergenceSetByUser())
        return false;

    return _useMaxForceConv.get();   // nothing specified -> default
}

/**
 * @brief get use rms force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseRMSForceConv() const
{
    if (_useRMSForceConv.isSetBySomeone())
        return _useRMSForceConv.get();

    if (_rmsForceConv.isSetByUser())
        return true;

    // some other criterion was requested explicitly -> only use those
    if (_anyConvergenceSetByUser())
        return false;

    return _useRMSForceConv.get();   // nothing specified -> default
}
