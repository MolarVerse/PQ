#include "settings.hpp"
#include "exceptions.hpp"

using namespace std;

void Settings::setTemperature(double temperature)
{
    if (temperature < 0)
        throw InputFileException("Temperature must be positive");
    else
        _temperature = make_pair(true, temperature);
}

void Settings::setRelaxationTime(double relaxationTime)
{
    if (relaxationTime < 0)
        throw InputFileException("Relaxation time must be positive");
    else
        _relaxationTime = make_pair(true, relaxationTime);
}

void Settings::setTauManostat(double tau)
{
    if (tau < 0)
        throw InputFileException("Relaxation time of manostat must be positive");
    else
        _tauManostat = make_pair(true, tau);
}
