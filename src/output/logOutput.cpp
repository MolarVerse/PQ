#include "logOutput.hpp"

#include "outputMessages.hpp"   // for initialMomentumMessage

#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <string>    // for char_traits, operator<<

using namespace output;

/**
 * @brief write header title
 *
 * @return string
 */
void LogOutput::writeHeader() { _fp << header() << '\n' << std::flush; }

/**
 * @brief write a message to the log file if the simulation ended normally
 *
 */
void LogOutput::writeEndedNormally() { _fp << endedNormally() << '\n' << std::flush; }

/**
 * @brief write a warning message to the log file if density and box dimensions are set
 *
 */
void LogOutput::writeDensityWarning()
{
    _fp << "WARNING: Density and box dimensions set. Density will be ignored." << '\n' << std::flush;
}

/**
 * @brief write initial momentum to log file
 *
 * @param momentum
 */
void LogOutput::writeInitialMomentum(const double momentum)
{
    _fp << '\n' << std::flush;
    _fp << initialMomentumMessage(momentum) << '\n' << std::flush;
}