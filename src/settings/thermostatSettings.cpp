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
#include "timingsSettings.hpp"

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

void KineticsSettings::finalize()
{
    if (_endTemperature.isSetByUser() && _targetTemperature.isSetByUser())
    {
        throw exc::InputFileException(
            "Both end temperature and target temperature are set by the user. "
            "Only one should be defined."
        );
    }

    if (!_endTemperature.isSetByUser() && !_targetTemperature.isSetByUser())
    {
        throw exc::InputFileException(
            "Neither end temperature nor target temperature is set by the "
            "user. "
            "At least one must be defined."
        );
    }

    if (_endTemperature.isSetByUser() && !_targetTemperature.isSetByUser())
    {
        _targetTemperature.setBySystem(_endTemperature.get());
    }

    if (!_endTemperature.isSetByUser() && _targetTemperature.isSetByUser())
    {
        _endTemperature.setBySystem(_targetTemperature.get());
    }

    /*************************************************************************
     * If the start temperature is defined, the temperature ramp is enabled.
     **
     *************************************************************************/

    if (!_startTemperature.isSetByUser())
    {
        // TODO: set values accordingly
        return;
    }

    /*************************************************************
     * If steps is 0, set the steps to the total number of steps *
     *************************************************************/

    auto       steps             = _temperatureRampSteps.get();
    const auto useFullSimulation = steps == 0;

    if (useFullSimulation)
        steps = settings::TimingsSettings::getNumberOfSteps();

    if (steps == 0)
        throw exc::InputFileException(
            "Temperature ramp requires at least one simulation step"
        );

    const auto frequency = _temperatureRampFrequency.get();

    if (frequency == 0)
        throw exc::InputFileException(
            "Temperature ramp frequency must be greater than zero"
        );

    if (useFullSimulation)
        _temperatureRampSteps.setBySystem(steps);

    /*************************************************************
     * resetting the target temperature to the start temperature *
     *************************************************************/

    const auto startTemp = _startTemperature.get();

    const auto targetTemp   = _targetTemperature.get();
    const auto tempDelta    = targetTemp - startTemp;
    const auto remainder    = steps % frequency == 0 ? 0U : 1U;
    const auto updates      = (steps / frequency) + remainder;
    const auto tempIncrease = tempDelta / static_cast<double>(updates);

    _temperatureIncrease.setBySystem(tempIncrease);
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
 * @brief set the temperature ramp frequency
 *
 * @param frequency
 */
void KineticsSettings::setRampingFrequencyByUser(size_t frequency)
{
    _temperatureRampFrequency.setByUser(frequency);
}

/**
 * @brief set the temperature ramp steps
 *
 * @param steps
 */
void KineticsSettings::setRampingStepsByUser(size_t steps)
{
    _temperatureRampSteps.setByUser(steps);
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
 * @brief get the temperature ramping steps
 *
 */
size_t KineticsSettings::getRampingSteps() const
{
    return _temperatureRampSteps.get();
}

/**
 * @brief get the temperature ramping frequency
 *
 * @return size_t
 */
size_t KineticsSettings::getRampingFrequency() const
{
    return _temperatureRampFrequency.get();
}

/**
 * @brief get the temperature increase
 *
 * @return const TemperatureDiff&
 */
const TemperatureDiff& KineticsSettings::getTemperatureIncrease() const
{
    return _temperatureIncrease.get();
}

/**
 * @brief get the start temperature
 *
 * @return const Temperature&
 */
const Temperature& KineticsSettings::getStartTemperature() const
{
    return _startTemperature.get();
}

/**
 * @brief get the end temperature
 *
 * @return const Temperature&
 */
const Temperature& KineticsSettings::getEndTemperature() const
{
    return _endTemperature.get();
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

/**
 * @brief check if the start temperature was set by the user, i.e. if a
 * temperature ramp is requested
 *
 * @return true if a start temperature was given
 */
bool KineticsSettings::isStartTemperatureSet() const
{
    return _startTemperature.isSetByUser();
}

/**
 * @brief set the start temperature by user
 *
 * @param startTemperature the start temperature to set
 */
void KineticsSettings::setStartTemperatureByUser(
    const Temperature& startTemperature
)
{
    _startTemperature.setByUser(startTemperature);
}

/**
 * @brief set the end temperature by user
 *
 * @param endTemperature the end temperature to set
 */
void KineticsSettings::setEndTemperatureByUser(
    const Temperature& endTemperature
)
{
    _endTemperature.setByUser(endTemperature);
}

/**
 * @brief set the target temperature by user
 *
 * @param targetTemperature the target temperature to set
 */
void KineticsSettings::setTargetTemperatureByUser(
    const Temperature& targetTemperature
)
{
    _targetTemperature.setByUser(targetTemperature);
}
