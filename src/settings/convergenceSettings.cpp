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
void ConvSettings::setEnergyConvergence(const double energyConv)
{
    _energyConvergence = energyConv;
}

/**
 * @brief set relative energy convergence
 *
 * @param relEnergyConv
 */
void ConvSettings::setRelEnergyConvergence(const double relEnergyConv)
{
    _relEnergyConvergence = relEnergyConv;
}

/**
 * @brief set absolute energy convergence
 *
 * @param absEnergyConv
 */
void ConvSettings::setAbsEnergyConvergence(const double absEnergyConv)
{
    _absEnergyConvergence = absEnergyConv;
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
void ConvSettings::setForceConvergence(const double forceConv)
{
    _forceConvergence = forceConv;
}

/**
 * @brief set relative force convergence
 *
 * @details this method is used to set the relative force convergence for the
 * max force and the rms force convergence
 *
 * @param relForceConv
 */
void ConvSettings::setRelForceConvergence(const double relForceConv)
{
    _relForceConvergence = relForceConv;
}

/**
 * @brief set absolute force convergence
 *
 * @details this method is used to set the absolute force convergence for the
 * max force and the rms force convergence
 *
 * @param absForceConv
 */
void ConvSettings::setAbsForceConvergence(const double absForceConv)
{
    _absForceConvergence = absForceConv;
}

/**
 * @brief set max force convergence
 *
 * @details this method is used to set the max force convergence
 * for the absolute and relative convergence
 *
 * @param maxForceConv
 */
void ConvSettings::setMaxForceConvergence(const double maxForceConv)
{
    _maxForceConvergence = maxForceConv;
}

/**
 * @brief set absolute max force convergence
 *
 * @param absMaxForceConv
 */
void ConvSettings::setAbsMaxForceConvergence(const double absMaxForceConv)
{
    _absMaxForceConvergence = absMaxForceConv;
}

/**
 * @brief set relative max force convergence
 *
 * @param relMaxForceConv
 */
void ConvSettings::setRelMaxForceConvergence(const double relMaxForceConv)
{
    _relMaxForceConvergence = relMaxForceConv;
}

/**
 * @brief set relative max force convergence
 *
 * @details this method is used to set the rms force convergence
 * for the absolute and relative convergence
 *
 * @param relMaxForceConv
 */
void ConvSettings::setRMSForceConvergence(const double rmsForceConv)
{
    _rmsForceConvergence = rmsForceConv;
}

/**
 * @brief set absolute rms force convergence
 *
 * @param absRMSForceConv
 */
void ConvSettings::setAbsRMSForceConvergence(const double absRMSForceConv)
{
    _absRMSForceConvergence = absRMSForceConv;
}

/**
 * @brief set relative rms force convergence
 *
 * @param relRMSForceConv
 */
void ConvSettings::setRelRMSForceConvergence(const double relRMSForceConv)
{
    _relRMSForceConvergence = relRMSForceConv;
}

/**
 * @brief set use energy convergence
 *
 * @param useEnergyConvergence
 */
void ConvSettings::setUseEnergyConvergence(const bool useEnergyConvergence)
{
    _useEnergyConvergence = useEnergyConvergence;
}

/**
 * @brief set use force convergence
 *
 * @param useForceConvergence
 */
void ConvSettings::setUseForceConvergence(const bool useForceConvergence)
{
    _useForceConvergence = useForceConvergence;
}

/**
 * @brief set use max force convergence
 *
 * @param useMaxForceConvergence
 */
void ConvSettings::setUseMaxForceConvergence(const bool useMaxForceConvergence)
{
    _useMaxForceConvergence = useMaxForceConvergence;
}

/**
 * @brief set use rms force convergence
 *
 * @param useRMSForceConvergence
 */
void ConvSettings::setUseRMSForceConvergence(const bool useRMSForceConvergence)
{
    _useRMSForceConvergence = useRMSForceConvergence;
}

/**
 * @brief set convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setConvergenceStrategy(const ConvStrategy strategy)
{
    _convergenceStrategy = strategy;
}

/**
 * @brief set convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setConvergenceStrategy(const std::string_view &strategy)
{
    _convergenceStrategy = determineConvStrategy(strategy);
}

/**
 * @brief set energy convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setEnergyConvergenceStrategy(const ConvStrategy strategy)
{
    _energyConvergenceStrategy = strategy;
}

/**
 * @brief set energy convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setEnergyConvergenceStrategy(const std::string_view &strategy
)
{
    _energyConvergenceStrategy = determineConvStrategy(strategy);
}

/**
 * @brief set force convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setForceConvergenceStrategy(const ConvStrategy strategy)
{
    _forceConvergenceStrategy = strategy;
}

/**
 * @brief set force convergence strategy
 *
 * @param strategy
 */
void ConvSettings::setForceConvergenceStrategy(const std::string_view &strategy)
{
    _forceConvergenceStrategy = determineConvStrategy(strategy);
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
std::optional<double> ConvSettings::getEnergyConvergence()
{
    return _energyConvergence;
}

/**
 * @brief get relative energy convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelEnergyConvergence()
{
    return _relEnergyConvergence;
}

/**
 * @brief get absolute energy convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsEnergyConvergence()
{
    return _absEnergyConvergence;
}

/**
 * @brief get force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getForceConvergence()
{
    return _forceConvergence;
}

/**
 * @brief get relative force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelForceConvergence()
{
    return _relForceConvergence;
}

/**
 * @brief get absolute force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsForceConvergence()
{
    return _absForceConvergence;
}

/**
 * @brief get max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getMaxForceConvergence()
{
    return _maxForceConvergence;
}

/**
 * @brief get absolute max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsMaxForceConvergence()
{
    return _absMaxForceConvergence;
}

/**
 * @brief get relative max force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelMaxForceConvergence()
{
    return _relMaxForceConvergence;
}

/**
 * @brief get rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRMSForceConvergence()
{
    return _rmsForceConvergence;
}

/**
 * @brief get absolute rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getAbsRMSForceConvergence()
{
    return _absRMSForceConvergence;
}

/**
 * @brief get relative rms force convergence
 *
 * @return std::optional<double>
 */
std::optional<double> ConvSettings::getRelRMSForceConvergence()
{
    return _relRMSForceConvergence;
}

/**
 * @brief get use energy convergence
 *
 * @return bool
 */
bool ConvSettings::getUseEnergyConvergence() { return _useEnergyConvergence; }

/**
 * @brief get use force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseForceConvergence() { return _useForceConvergence; }

/**
 * @brief get use max force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseMaxForceConvergence()
{
    return _useMaxForceConvergence;
}

/**
 * @brief get use rms force convergence
 *
 * @return bool
 */
bool ConvSettings::getUseRMSForceConvergence()
{
    return _useRMSForceConvergence;
}

/**
 * @brief get convergence strategy
 *
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getConvergenceStrategy()
{
    return _convergenceStrategy;
}

/**
 * @brief get energy convergence strategy
 *
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getEnergyConvergenceStrategy()
{
    return _energyConvergenceStrategy;
}

/**
 * @brief get force convergence strategy
 *
 * @return ConvStrategy
 */
ConvStrategy ConvSettings::getForceConvergenceStrategy()
{
    return _forceConvergenceStrategy;
}