#include "stdoutOutput.hpp"

#include "exceptions.hpp"       // for UserInputExceptionWarning, customException
#include "outputMessages.hpp"   // for initialMomentumMessage

#include <iostream>      // for operator<<, char_traits, basic_ostream, cout
#include <string>        // for operator<<
#include <string_view>   // for string_view

using namespace output;

/**
 * @brief write header title
 *
 * @return string
 */
void StdoutOutput::writeHeader() const { std::cout << header() << '\n' << std::flush; }

/**
 * @brief write a warning message to the stdout if density and box dimensions are set
 *
 */
void StdoutOutput::writeDensityWarning() const
{
    try
    {
        throw customException::UserInputExceptionWarning("Density and box dimensions set. Density will be ignored.");
    }
    catch (const customException::UserInputExceptionWarning &e)
    {
        std::cout << e.what() << '\n' << '\n' << std::flush;
    }
}

/**
 * @brief write initial momentum to stdout
 *
 * @param momentum
 */
void StdoutOutput::writeInitialMomentum(const double momentum) const
{
    std::cout << '\n';
    std::cout << initialMomentumMessage(momentum) << '\n' << std::flush;
}