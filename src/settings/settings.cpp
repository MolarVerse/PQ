#include <stdexcept>
#include <string>

#include "settings.hpp"

using namespace std;

int Settings::getStepCount() const { return _stepcount; }

/**
 * @brief Sets the step count of the simulation
 * 
 * @param stepcount 
 * 
 * @throw range_error if stepcount is negative
 */
void Settings::setStepCount(int stepcount)
{
    if(stepcount < 0) throw range_error("Step count in restart file must be positive - stepcount = " + to_string(stepcount) );
    _stepcount = stepcount;
}