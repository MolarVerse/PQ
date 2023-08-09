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
        cout << e.what() << '\n' << '\n' << flush;
    }
}

/**
 * @brief write initial momentum to stdout
 *
 * @param momentum
 */
void StdoutOutput::writeInitialMomentum(const double momentum) const
{
    cout << '\n';
    cout << initialMomentumMessage(momentum) << '\n';
}