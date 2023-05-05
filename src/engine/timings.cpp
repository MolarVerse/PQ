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

/**
 * @brief Sets the number of steps of the simulation
 * 
 * @param numberOfSteps 
 * 
 * @throw range_error if number of steps is negative
 */
void Timings::setNumberOfSteps(int numberOfSteps)
{
    if(numberOfSteps < 0) throw range_error("Number of steps in restart file must be positive - nstep = " + to_string(numberOfSteps) );
    _numberOfSteps = numberOfSteps;
}