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

#include "thermostatSettings.hpp"

#include "stringUtilities.hpp"   // for toLowerCopy

using namespace settings;
using namespace utilities;

/**
 * @brief return string of thermostatType
 *
 * @param thermostatType
 * @return std::string
 */
std::string settings::string(const ThermostatType &thermostatType)
{
    switch (thermostatType)
    {
        using enum ThermostatType;

        case BERENDSEN: return "berendsen";
        case VELOCITY_RESCALING: return "velocity_rescaling";
        case LANGEVIN: return "langevin";
        case NOSE_HOOVER: return "nh-chain";

        default: return "none";
    }
}

/**
 * @brief add chi to the index map
 *
 * @param index
 * @param chi
 * @return auto
 */
auto ThermostatSettings::addChi(const size_t index, const double chi)
    -> decltype(_chi.try_emplace(index, chi))
{
    return _chi.try_emplace(index, chi);
}

/**
 * @brief add zeta to the index map
 *
 * @param index
 * @param zeta
 * @return auto
 */
auto ThermostatSettings::addZeta(const size_t index, const double zeta)
    -> decltype(_zeta.try_emplace(index, zeta))
{
    return _zeta.try_emplace(index, zeta);
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the thermostatType to enum in settings
 *
 * @param thermostatType
 */
void ThermostatSettings::setThermostatType(
    const std::string_view &thermostatType
)
{
    using enum ThermostatType;
    const auto thermostatTypeToLower =
        toLowerAndReplaceDashesCopy(thermostatType);

    if (thermostatTypeToLower == "berendsen")
        _thermostatType = BERENDSEN;

    else if (thermostatTypeToLower == "velocity_rescaling")
        _thermostatType = VELOCITY_RESCALING;

    else if (thermostatTypeToLower == "langevin")
        _thermostatType = LANGEVIN;

    else if (thermostatTypeToLower == "nh_chain")
        _thermostatType = NOSE_HOOVER;

    else
        _thermostatType = NONE;
}

/**
 * @brief sets the thermostatType to enum in settings
 *
 * @param thermostatType
 */
void ThermostatSettings::setThermostatType(const ThermostatType &thermostatType)
{
    _thermostatType = thermostatType;
}

/**
 * @brief set the nose hoover chain length
 *
 * @param length
 */
void ThermostatSettings::setNoseHooverChainLength(const size_t length)
{
    _nhChainLength = length;
}

/**
 * @brief set the temperature ramp steps
 *
 * @param steps
 */
void ThermostatSettings::setTemperatureRampSteps(const size_t steps)
{
    _temperatureRampSteps = steps;
}

/**
 * @brief set the temperature ramp frequency
 *
 * @param frequency
 */
void ThermostatSettings::setTemperatureRampFrequency(const size_t frequency)
{
    _temperatureRampFrequency = frequency;
}

/**
 * @brief set the temperature set
 *
 * @param temperatureSet
 */
void ThermostatSettings::setTemperatureSet(const bool temperatureSet)
{
    _isTemperatureSet = temperatureSet;
}

/**
 * @brief set the start temperature set
 *
 * @param startTemperatureSet
 */
void ThermostatSettings::setStartTemperatureSet(const bool startTemperatureSet)
{
    _isStartTemperatureSet = startTemperatureSet;
}

/**
 * @brief set the end temperature set
 *
 * @param endTemperatureSet
 */
void ThermostatSettings::setEndTemperatureSet(const bool endTemperatureSet)
{
    _isEndTemperatureSet = endTemperatureSet;
}

/**
 * @brief set the target temperature
 *
 * @param targetTemperature
 */
void ThermostatSettings::setTargetTemperature(const double targetTemperature)
{
    _targetTemperature = targetTemperature;
    setTemperatureSet(true);
    setActualTargetTemperature(targetTemperature);
}

/**
 * @brief set the actual target temperature
 *
 * @param actualTargetTemperature
 */
void ThermostatSettings::setActualTargetTemperature(
    const double actualTargetTemperature
)
{
    _actualTargetTemperature = actualTargetTemperature;
}

/**
 * @brief set the start temperature
 *
 * @param startTemperature
 */
void ThermostatSettings::setStartTemperature(const double startTemperature)
{
    _startTemperature = startTemperature;
    setStartTemperatureSet(true);
}

/**
 * @brief set the end temperature
 *
 * @param endTemperature
 */
void ThermostatSettings::setEndTemperature(const double endTemperature)
{
    _endTemperature = endTemperature;
    setEndTemperatureSet(true);
}

/**
 * @brief set the relaxation time
 *
 * @param relaxationTime
 */
void ThermostatSettings::setRelaxationTime(const double relaxationTime)
{
    _relaxationTime = relaxationTime;
}

/**
 * @brief set the friction
 *
 * @param friction
 */
void ThermostatSettings::setFriction(const double friction)
{
    _friction = friction;
}

/**
 * @brief set the nose hoover coupling frequency
 *
 * @param frequency
 */
void ThermostatSettings::setNoseHooverCouplingFrequency(const double frequency)
{
    _nhCouplingFreq = frequency;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the nose hoover chain length
 *
 * @return size_t
 */
size_t ThermostatSettings::getNoseHooverChainLength() { return _nhChainLength; }

/**
 * @brief get the temperature ramp steps
 *
 * @return size_t
 */
size_t ThermostatSettings::getTemperatureRampSteps()
{
    return _temperatureRampSteps;
}

/**
 * @brief get the temperature ramp frequency
 *
 * @return size_t
 */
size_t ThermostatSettings::getTemperatureRampFrequency()
{
    return _temperatureRampFrequency;
}

/**
 * @brief get the thermostat type
 *
 * @return ThermostatType
 */
ThermostatType ThermostatSettings::getThermostatType()
{
    return _thermostatType;
}

/**
 * @brief is the temperature set
 *
 * @return bool
 */
bool ThermostatSettings::isTemperatureSet() { return _isTemperatureSet; }

/**
 * @brief is the start temperature set
 *
 * @return bool
 */
bool ThermostatSettings::isStartTemperatureSet()
{
    return _isStartTemperatureSet;
}

/**
 * @brief is the end temperature set
 *
 * @return bool
 */
bool ThermostatSettings::isEndTemperatureSet() { return _isEndTemperatureSet; }

/**
 * @brief get the target temperature
 *
 * @return double
 */
double ThermostatSettings::getTargetTemperature() { return _targetTemperature; }

/**
 * @brief get the actual target temperature
 *
 * @return double
 */
double ThermostatSettings::getActualTargetTemperature()
{
    return _actualTargetTemperature;
}

/**
 * @brief get the start temperature
 *
 * @return double
 */
double ThermostatSettings::getStartTemperature() { return _startTemperature; }

/**
 * @brief get the end temperature
 *
 * @return double
 */
double ThermostatSettings::getEndTemperature() { return _endTemperature; }

/**
 * @brief get the relaxation time
 *
 * @return double
 */
double ThermostatSettings::getRelaxationTime() { return _relaxationTime; }

/**
 * @brief get the friction
 *
 * @return double
 */
double ThermostatSettings::getFriction() { return _friction; }

/**
 * @brief get the nose hoover coupling frequency
 *
 * @return double
 */
double ThermostatSettings::getNoseHooverCouplingFrequency()
{
    return _nhCouplingFreq;
}

/**
 * @brief get chi
 *
 * @return std::map<size_t, double>
 */
std::map<size_t, double> ThermostatSettings::getChi() { return _chi; }

/**
 * @brief get zeta
 *
 * @return std::map<size_t, double>
 */
std::map<size_t, double> ThermostatSettings::getZeta() { return _zeta; }