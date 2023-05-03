#include <stdexcept>

#include "timings.hpp"

using namespace std;

/**
 * @brief Sets the step count of the simulation
 * 
 * @param stepCount 
 * 
 * @throw range_error if stepcount is negative
 */
void Timings::setStepCount(int stepCount)
{
    if(stepCount < 0) throw range_error("Step count in restart file must be positive - step count = " + to_string(stepCount) );
    _stepCount = stepCount;
}