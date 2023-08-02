#include "stdoutOutput.hpp"

#include "exceptions.hpp"

using namespace std;
using namespace output;
using namespace customException;

/**
 * @brief write a warning message to the stdout if density and box dimensions are set
 *
 */
void StdoutOutput::writeDensityWarning() const
{
    try
    {
        throw UserInputExceptionWarning("Density and box dimensions set. Density will be ignored.");
    }
    catch (const UserInputExceptionWarning &e)
    {
        cout << e.what() << endl << endl;
    }
}

/**
 * @brief write initial momentum to stdout
 *
 * @param momentum
 */
void StdoutOutput::writeInitialMomentum(const double momentum) const
{
    cout << endl;
    cout << initialMomentumMessage(momentum) << endl;
}

/**
 * @brief write warning message to stdout if Berendsen thermostat is set but no relaxation time is given
 *
 */
void StdoutOutput::writeRelaxationTimeThermostatWarning() const
{
    try
    {
        throw UserInputExceptionWarning("Berendsen thermostat set but no relaxation time given. Using default value of 0.1ps.");
    }
    catch (const UserInputExceptionWarning &e)
    {
        cout << e.what() << endl << endl;
    }
}