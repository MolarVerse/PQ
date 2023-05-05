#include "timings.hpp"
#include "exceptions.hpp"

using namespace std;

/**
 * @brief Sets the step count of the simulation
 *
 * @param stepCount
 *
 * @throw InputFileException if stepcount is negative
 */
void Timings::setStepCount(int stepCount)
{
    if (stepCount < 0)
        throw InputFileException("Step count in restart file must be positive - step count = " + to_string(stepCount));
    _stepCount = stepCount;
}

/**
 * @brief Sets the number of steps of the simulation
 *
 * @param numberOfSteps
 *
 * @throw InputFileException if number of steps is negative
 */
void Timings::setNumberOfSteps(int numberOfSteps)
{
    if (numberOfSteps < 0)
        throw InputFileException("Number of steps in restart file must be positive - nstep = " + to_string(numberOfSteps));
    _numberOfSteps = numberOfSteps;
}