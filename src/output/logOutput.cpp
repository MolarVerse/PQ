#include "logOutput.hpp"

#include "outputMessages.hpp"   // for initialMomentumMessage

#include <format>    // for format
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
void LogOutput::writeEndedNormally(const double elapsedTime)
{
    _fp << elapsedTimeMessage(elapsedTime) << '\n';
    _fp << endedNormally() << '\n' << std::flush;
}

/**
 * @brief write a warning message to the log file if density and box dimensions are set
 *
 */
void LogOutput::writeDensityWarning()
{
    _fp << _WARNING_ << "Density and box dimensions set. Density will be ignored." << '\n' << std::flush;
}

/**
 * @brief write initial momentum to log file
 *
 * @param momentum
 */
void LogOutput::writeInitialMomentum(const double momentum)
{
    _fp << std::format("\n{}Initial momentum = {} Angstrom * amu / fs\n", _INFO_, momentum) << std::flush;
}