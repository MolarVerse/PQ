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
ConvStrategy ConvSettings::determineConvStrategy(
    const std::string_view &strategy
)
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
 * @brief set relative force convergence
 *
 * @details this method is used to set the relative force convergence for the
 * max force and the rms force convergence
 *
 * @param relForceConv
 */
void ConvSettings::setRelForceConv(const double relForceConv)
{
    _relForceConv = relForceConv;
}

/**
 * @brief set absolute force convergence
 *
 * @details this method is used to set the absolute force convergence for the
 * max force and the rms force convergence
 *
 * @param absForceConv
 */
void ConvSettings::setAbsForceConv(const double absForceConv)
{
    _absForceConv = absForceConv;
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
 * @brief set absolute max force convergence
 *
 * @param absMaxForceConv
 */
void ConvSettings::setAbsMaxForceConv(const double absMaxForceConv)
{
    _absMaxForceConv = absMaxForceConv;
}

/**
 * @brief set relative max force convergence
 *
 * @param relMaxForceConv
 */
void ConvSettings::setRelMaxForceConv(const double relMaxForceConv)
{
    _relMaxForceConv = relMaxForceConv;
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
 * @brief set absolute rms force convergence
 *
 * @param absRMSForceConv
 */
void ConvSettings::setAbsRMSForceConv(const double absRMSForceConv)
{
    _absRMSForceConv = absRMSForceConv;
}

/**
 * @brief set relative rms force convergence
 *
 * @param relRMSForceConv
 */
void ConvSettings::setRelRMSForceConv(const double relRMSForceConv)
{
    _relRMSForceConv = relRMSForceConv;
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
 * @brief set convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setConvStrategy(const ConvStrategy strategy)
{
    _convStrategy = strategy;
}

/**
 * @brief set convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setConvStrategy(const std::string_view &strategy)
{
    _convStrategy = determineConvStrategy(strategy);
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
    _energyConvStrategy = determineConvStrategy(strategy);
}

/**
 * @brief set force convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setForceConvStrategy(const ConvStrategy strategy)
{
    _forceConvStrategy = strategy;
}

/**
 * @brief set force convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setForceConvStrategy(const std::string_view &strategy)
{
    _forceConvStrategy = determineConvStrategy(strategy);
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
 * @brief get relative force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelForceConv() { return _relForceConv; }

/**
 * @brief get absolute force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsForceConv() { return _absForceConv; }

/**
 * @brief get max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getMaxForceConv() { return _maxForceConv; }

/**
 * @brief get absolute max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsMaxForceConv()
{
    return _absMaxForceConv;
}

/**
 * @brief get relative max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelMaxForceConv()
{
    return _relMaxForceConv;
}

/**
 * @brief get rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRMSForceConv() { return _rmsForceConv; }

/**
 * @brief get absolute rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsRMSForceConv()
{
    return _absRMSForceConv;
}

/**
 * @brief get relative rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelRMSForceConv()
{
    return _relRMSForceConv;
}

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
 * @brief get convergence strategy
 *
 * @return ConvStrategy
 */
std::optional<ConvStrategy> ConvSettings::getConvStrategy()
{
    return _convStrategy;
}

/**
 * @brief get energy convergence strategy
 *
 * @return ConvStrategy
 */
std::optional<ConvStrategy> ConvSettings::getEnergyConvStrategy()
{
    return _energyConvStrategy;
}

/**
 * @brief get force convergence strategy
 *
 * @return ConvStrategy
 */
std::optional<ConvStrategy> ConvSettings::getForceConvStrategy()
{
    return _forceConvStrategy;
}

/**
 * @brief get default energy convergence strategy
 *
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getDefaultEnergyConvStrategy()
{
    return determineConvStrategy(_defaultEnergyConvStrategy);
}

/**
 * @brief get default force convergence strategy
 *
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getDefaultForceConvStrategy()
{
    return determineConvStrategy(_defaultForceConvStrategy);
}