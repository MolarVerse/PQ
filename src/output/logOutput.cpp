#include "logOutput.hpp"

#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <string>    // for char_traits, operator<<

using namespace std;
using namespace output;

/**
 * @brief write a warning message to the log file if density and box dimensions are set
 *
 */
void LogOutput::writeDensityWarning()
{
    _fp << "WARNING: Density and box dimensions set. Density will be ignored." << '\n' << flush;
}

/**
 * @brief write initial momentum to log file
 *
 * @param momentum
 */
void LogOutput::writeInitialMomentum(const double momentum)
{
    _fp << '\n' << flush;
    _fp << initialMomentumMessage(momentum) << '\n' << flush;
}