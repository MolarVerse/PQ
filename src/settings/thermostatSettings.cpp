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

#include "referenceSettings.hpp"

/**
 * @brief finalize the thermostat settings by checking if all required
 * temperatures are set
 *
 * @param referenceSettings
 */
void ThermostatSettings::finalize(ReferenceSettings& referenceSettings)
{
    switch (_thermostatType.get())
    {
        case ThermostatType::BERENDSEN:
            referenceSettings.addReferenceFile(ReferenceFiles::BERENDSEN_FILE);
            break;
        case ThermostatType::VELOCITY_RESCALING:
            referenceSettings.addReferenceFile(
                ReferenceFiles::VELOCITY_RESCALING_FILE
            );
            break;
        case ThermostatType::LANGEVIN:
            referenceSettings.addReferenceFile(ReferenceFiles::LANGEVIN_FILE);
            break;
        case ThermostatType::NOSE_HOOVER:
            referenceSettings.addReferenceFile(
                ReferenceFiles::NOSE_HOOVER_CHAIN_FILE
            );
            break;
        case ThermostatType::NONE: break;
    }
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
void ThermostatSettings::setThermostatTypeByUser(ThermostatType thermostatType)
{
    _thermostatType.setByUser(thermostatType);
}

/**
 * @brief set the nose hoover chain length
 *
 * @param length
 */
void NoseHooverSettings::setNoseHooverChainLengthByUser(size_t length)
{
    _nhChainLength.setByUser(length);
}

/**
 * @brief set the temperature ramp steps
 *
 * @param steps
 */
void ThermostatSettings::setTemperatureRampSteps(size_t steps)
{
    _temperatureRampSteps = steps;
}

/**
 * @brief set the temperature ramp frequency
 *
 * @param frequency
 */
void ThermostatSettings::setTemperatureRampFrequency(size_t frequency)
{
    _temperatureRampFrequency = frequency;
}

/**
 * @brief set the temperature set
 *
 * @param temperatureSet
 */
void ThermostatSettings::setTemperatureSet(bool temperatureSet)
{
    _isTemperatureSet = temperatureSet;
}

/**
 * @brief set the start temperature set
 *
 * @param startTemperatureSet
 */
void ThermostatSettings::setStartTemperatureSet(bool startTemperatureSet)
{
    _isStartTemperatureSet = startTemperatureSet;
}

/**
 * @brief set the end temperature set
 *
 * @param endTemperatureSet
 */
void ThermostatSettings::setEndTemperatureSet(bool endTemperatureSet)
{
    _isEndTemperatureSet = endTemperatureSet;
}

/**
 * @brief set the target temperature
 *
 * @param targetTemperature
 */
void ThermostatSettings::setTargetTemperature(
    const Temperature& targetTemperature
)
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
    const Temperature& actualTargetTemperature
)
{
    _actualTargetTemperature = actualTargetTemperature;
}

/**
 * @brief set the start temperature
 *
 * @param startTemperature
 */
void ThermostatSettings::setStartTemperature(
    const Temperature& startTemperature
)
{
    _startTemperature = startTemperature;
    setStartTemperatureSet(true);
}

/**
 * @brief set the end temperature
 *
 * @param endTemperature
 */
void ThermostatSettings::setEndTemperature(const Temperature& endTemperature)
{
    _endTemperature = endTemperature;
    setEndTemperatureSet(true);
}

/**
 * @brief set the relaxation time
 *
 * @param relaxationTime
 */
void ThermostatSettings::setRelaxationTime(const Time& relaxationTime)
{
    _relaxationTime = relaxationTime;
}

/**
 * @brief set the friction
 *
 * @param friction
 */
void ThermostatSettings::setFriction(double friction) { _friction = friction; }

/**
 * @brief set the nose hoover coupling frequency
 *
 * @param frequency
 */
void NoseHooverSettings::setNoseHooverCouplingFrequencyByUser(
    const Frequency& frequency
)
{
    _nhCouplingFreq.setByUser(frequency);
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
size_t NoseHooverSettings::getNoseHooverChainLength() const
{
    return _nhChainLength.get();
}

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
ThermostatType ThermostatSettings::getThermostatType() const
{
    return _thermostatType.get();
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
 * @return const Temperature&
 */
const Temperature& ThermostatSettings::getTargetTemperature()
{
    return _targetTemperature;
}

/**
 * @brief get the actual target temperature
 *
 * @return const Temperature&
 */
const Temperature& ThermostatSettings::getActualTargetTemperature()
{
    return _actualTargetTemperature;
}

/**
 * @brief get the start temperature
 *
 * @return const Temperature&
 */
const Temperature& ThermostatSettings::getStartTemperature()
{
    return _startTemperature;
}

/**
 * @brief get the end temperature
 *
 * @return const Temperature&
 */
const Temperature& ThermostatSettings::getEndTemperature()
{
    return _endTemperature;
}

/**
 * @brief get the relaxation time
 *
 * @return const Time&
 */
const Time& ThermostatSettings::getRelaxationTime() { return _relaxationTime; }

/**
 * @brief get the friction
 *
 * @return double
 */
double ThermostatSettings::getFriction() { return _friction; }

/**
 * @brief get the nose hoover coupling frequency
 *
 * @return const Frequency&
 */
const Frequency& NoseHooverSettings::getNoseHooverCouplingFrequency() const
{
    return _nhCouplingFreq.get();
}

/**
 * @brief get chi
 *
 * @return std::map<size_t, double>
 */
const std::map<size_t, double>& NoseHooverSettings::getChi() const
{
    return _chi.get();
}

/**
 * @brief get zeta
 *
 * @return std::map<size_t, double>
 */
const std::map<size_t, double>& NoseHooverSettings::getZeta() const
{
    return _zeta.get();
}

/**
 * @brief check if both chi and zeta are set
 *
 * @return bool
 */
bool NoseHooverSettings::isChiAndZetaSet() const
{
    return _chi.isSetBySomeone() && _zeta.isSetBySomeone();
}

/**
 * @brief set chi by user
 *
 * @param chi the chi values to set
 */
void NoseHooverSettings::setChiByUser(const std::map<size_t, double>& chi)
{
    _chi.setByUser(chi);
}

/**
 * @brief set zeta by user
 *
 * @param zeta the zeta values to set
 */
void NoseHooverSettings::setZetaByUser(const std::map<size_t, double>& zeta)
{
    _zeta.setByUser(zeta);
}
