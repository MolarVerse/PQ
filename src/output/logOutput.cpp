#include "logOutput.hpp"

using namespace std;
using namespace output;

/**
 * @brief write a warning message to the log file if density and box dimensions are set
 *
 */
void LogOutput::writeDensityWarning() { _fp << "WARNING: Density and box dimensions set. Density will be ignored." << endl; }

/**
 * @brief write initial momentum to log file
 *
 * @param momentum
 */
void LogOutput::writeInitialMomentum(const double momentum)
{
    _fp << endl;
    _fp << initialMomentumMessage(momentum) << endl;
}