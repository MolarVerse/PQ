#include "settings.hpp"

#include "exceptions.hpp"

using namespace std;
using namespace settings;
using namespace customException;

/**
 * @brief set temperature
 *
 * @param temperature
 *
 * @throw InputFileException if temperature is negative
 */
void Settings::setTemperature(double temperature)
{
    if (temperature < 0)
        throw InputFileException("Temperature must be positive");
    else
        _temperature = make_pair(true, temperature);
}

/**
 * @brief set relaxation time for thermostat
 *
 * @param relaxationTime
 *
 * @throw InputFileException if relaxation time is negative
 */
void Settings::setRelaxationTime(double relaxationTime)
{
    if (relaxationTime < 0)
        throw InputFileException("Relaxation time must be positive");
    else
        _relaxationTime = make_pair(true, relaxationTime);
}

/**
 * @brief set relaxation time for manostat
 *
 * @param tau
 *
 * @throw InputFileException if relaxation time is negative
 */
void Settings::setTauManostat(double tau)
{
    if (tau < 0)
        throw InputFileException("Relaxation time of manostat must be positive");
    else
        _tauManostat = make_pair(true, tau);
}
