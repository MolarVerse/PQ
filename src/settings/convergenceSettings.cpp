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

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set energy convergence
 *
 * @details this method is used to set both the energy convergence for the
 * absolute and relative convergence
 *
 * @param energyConv
 */
void ConvSettings::setEnergyConv(double energyConv)
{
    _energyConv.set(energyConv);
}

/**
 * @brief set relative energy convergence
 *
 * @param relEnergyConv
 */
void ConvSettings::setRelEnergyConv(double relEnergyConv)
{
    _relEnergyConv.set(relEnergyConv);
}

/**
 * @brief set absolute energy convergence
 *
 * @param absEnergyConv
 */
void ConvSettings::setAbsEnergyConv(double absEnergyConv)
{
    _absEnergyConv.set(absEnergyConv);
}

/**
 * @brief set force convergence
 *
 * @details this method is used to set both the force convergence for the
 * absolute and relative convergence as well as to set them both for the
 * max force and the rms force convergence
 *
 * @param forceConv
 */
void ConvSettings::setForceConv(double forceConv) { _forceConv.set(forceConv); }

/**
 * @brief set max force convergence
 *
 * @details this method is used to set the max force convergence
 * for the absolute and relative convergence
 *
 * @param maxForceConv
 */
void ConvSettings::setMaxForceConv(double maxForceConv)
{
    _maxForceConv.set(maxForceConv);
}

/**
 * @brief set relative max force convergence
 *
 * @details this method is used to set the rms force convergence
 * for the absolute and relative convergence
 *
 * @param rmsForceConv
 */
void ConvSettings::setRMSForceConv(double rmsForceConv)
{
    _rmsForceConv.set(rmsForceConv);
}

/**
 * @brief set use energy convergence
 *
 * @param useEnergyConvergence
 */
void ConvSettings::setUseEnergyConv(bool useEnergyConvergence)
{
    _useEnergyConv.set(useEnergyConvergence);
}

/**
 * @brief set use force convergence
 *
 * @param useForceConvergence
 */
void ConvSettings::setUseForceConv(bool useForceConvergence)
{
    _useForceConv.set(useForceConvergence);
}

/**
 * @brief set use max force convergence
 *
 * @param useMaxForceConvergence
 */
void ConvSettings::setUseMaxForceConv(bool useMaxForceConvergence)
{
    _useMaxForceConv.set(useMaxForceConvergence);
}

/**
 * @brief set use rms force convergence
 *
 * @param useRMSForceConvergence
 */
void ConvSettings::setUseRMSForceConv(bool useRMSForceConvergence)
{
    _useRMSForceConv.set(useRMSForceConvergence);
}

/**
 * @brief set energy convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setEnergyConvStrategy(ConvStrategy strategy)
{
    _energyConvStrategy.set(strategy);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get energy convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getEnergyConv() const
{
    if (!_energyConv.isSet())
        return std::nullopt;

    return _energyConv.get();
}

/**
 * @brief get relative energy convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelEnergyConv() const
{
    if (!_relEnergyConv.isSet())
        return std::nullopt;

    return _relEnergyConv.get();
}

/**
 * @brief get absolute energy convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsEnergyConv() const
{
    if (!_absEnergyConv.isSet())
        return std::nullopt;

    return _absEnergyConv.get();
}

/**
 * @brief get force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getForceConv() const
{
    if (!_forceConv.isSet())
        return std::nullopt;

    return _forceConv.get();
}

/**
 * @brief get max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getMaxForceConv() const
{
    if (!_maxForceConv.isSet())
        return std::nullopt;

    return _maxForceConv.get();
}

/**
 * @brief get rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRMSForceConv() const
{
    if (!_rmsForceConv.isSet())
        return std::nullopt;

    return _rmsForceConv.get();
}

/**
 * @brief get use energy convergence
 *
 * @return bool
 */
bool ConvSettings::getUseEnergyConv() const { return _useEnergyConv.get(); }

/**
 * @brief get use force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseForceConv() const { return _useForceConv.get(); }

/**
 * @brief get use max force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseMaxForceConv() const { return _useMaxForceConv.get(); }

/**
 * @brief get use rms force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseRMSForceConv() const { return _useRMSForceConv.get(); }

/**
 * @brief get energy convergence strategy
 *
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getEnConvStrategy() const
{
    return _energyConvStrategy.get();
}
