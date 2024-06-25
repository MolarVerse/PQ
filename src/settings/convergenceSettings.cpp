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

using namespace settings;

/**
 * @brief returns the convergence strategy as string
 *
 * @param strategy
 * @return std::string
 */
std::string settings::string(const ConvStrategy strategy)
{
    switch (strategy)
    {
        case ConvStrategy::RIGOROUS: return "RIGOROUS";

        case ConvStrategy::LOOSE: return "LOOSE";

        case ConvStrategy::ABSOLUTE: return "ABSOLUTE";

        case ConvStrategy::RELATIVE: return "RELATIVE";

        default: return "none";
    }
}

/**
 * @brief get ConvStrategy from string
 *
 * @param strategy
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getConvStrategy(const std::string_view &strategy)
{
    if ("rigorous" == strategy)
        return ConvStrategy::RIGOROUS;

    else if ("loose" == strategy)
        return ConvStrategy::LOOSE;

    else if ("absolute" == strategy)
        return ConvStrategy::ABSOLUTE;

    else if ("relative" == strategy)
        return ConvStrategy::RELATIVE;

    else
        return ConvStrategy::RIGOROUS;
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
void ConvSettings::setEnergyConv(const double energyConv)
{
    _energyConv = energyConv;
}

/**
 * @brief set relative energy convergence
 *
 * @param relEnergyConv
 */
void ConvSettings::setRelEnergyConv(const double relEnergyConv)
{
    _relEnergyConv = relEnergyConv;
}

/**
 * @brief set absolute energy convergence
 *
 * @param absEnergyConv
 */
void ConvSettings::setAbsEnergyConv(const double absEnergyConv)
{
    _absEnergyConv = absEnergyConv;
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
void ConvSettings::setForceConv(const double forceConv)
{
    _forceConv = forceConv;
}

/**
 * @brief set max force convergence
 *
 * @details this method is used to set the max force convergence
 * for the absolute and relative convergence
 *
 * @param maxForceConv
 */
void ConvSettings::setMaxForceConv(const double maxForceConv)
{
    _maxForceConv = maxForceConv;
}

/**
 * @brief set relative max force convergence
 *
 * @details this method is used to set the rms force convergence
 * for the absolute and relative convergence
 *
 * @param relMaxForceConv
 */
void ConvSettings::setRMSForceConv(const double rmsForceConv)
{
    _rmsForceConv = rmsForceConv;
}

/**
 * @brief set use energy convergence
 *
 * @param useEnergyConvergence
 */
void ConvSettings::setUseEnergyConv(const bool useEnergyConvergence)
{
    _useEnergyConv = useEnergyConvergence;
}

/**
 * @brief set use force convergence
 *
 * @param useForceConvergence
 */
void ConvSettings::setUseForceConv(const bool useForceConvergence)
{
    _useForceConv = useForceConvergence;
}

/**
 * @brief set use max force convergence
 *
 * @param useMaxForceConvergence
 */
void ConvSettings::setUseMaxForceConv(const bool useMaxForceConvergence)
{
    _useMaxForceConv = useMaxForceConvergence;
}

/**
 * @brief set use rms force convergence
 *
 * @param useRMSForceConvergence
 */
void ConvSettings::setUseRMSForceConv(const bool useRMSForceConvergence)
{
    _useRMSForceConv = useRMSForceConvergence;
}

/**
 * @brief set energy convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setEnergyConvStrategy(const ConvStrategy strategy)
{
    _energyConvStrategy = strategy;
}

/**
 * @brief set energy convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setEnergyConvStrategy(const std::string_view &strategy)
{
    _energyConvStrategy = getConvStrategy(strategy);
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
std::optional<double> ConvSettings::getEnergyConv() { return _energyConv; }

/**
 * @brief get relative energy convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelEnergyConv()
{
    return _relEnergyConv;
}

/**
 * @brief get absolute energy convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsEnergyConv()
{
    return _absEnergyConv;
}

/**
 * @brief get force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getForceConv() { return _forceConv; }

/**
 * @brief get max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getMaxForceConv() { return _maxForceConv; }

/**
 * @brief get rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRMSForceConv() { return _rmsForceConv; }

/**
 * @brief get use energy convergence
 *
 * @return bool
 */
bool ConvSettings::getUseEnergyConv() { return _useEnergyConv; }

/**
 * @brief get use force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseForceConv() { return _useForceConv; }

/**
 * @brief get use max force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseMaxForceConv() { return _useMaxForceConv; }

/**
 * @brief get use rms force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseRMSForceConv() { return _useRMSForceConv; }

/**
 * @brief get energy convergence strategy
 *
 * @return ConvStrategy
 */
std::optional<ConvStrategy> ConvSettings::getEnConvStrategy()
{
    return _energyConvStrategy;
}

/**
 * @brief get default energy convergence strategy
 *
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getDefaultEnergyConvStrategy()
{
    return getConvStrategy(_defaultEnergyConvStrategy);
}