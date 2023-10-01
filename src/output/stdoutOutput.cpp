/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "stdoutOutput.hpp"

#include "exceptions.hpp"       // for UserInputExceptionWarning, customException
#include "outputMessages.hpp"   // for initialMomentumMessage

#include <format>        // for format
#include <iostream>      // for operator<<, char_traits, basic_ostream, cout
#include <string>        // for operator<<
#include <string_view>   // for string_view

using output::StdoutOutput;

/**
 * @brief write header title
 *
 * @return string
 */
void StdoutOutput::writeHeader() const { std::cout << header() << '\n' << std::flush; }

/**
 * @brief write a message to the stdout if the simulation ended normally
 *
 */
void StdoutOutput::writeEndedNormally(const double elapsedTime) const
{
    std::cout << elapsedTimeMessage(elapsedTime) << '\n';
    std::cout << endedNormally() << '\n' << std::flush;
}

/**
 * @brief write a warning message to the stdout if density and box dimensions are set
 *
 */
void StdoutOutput::writeDensityWarning() const
{
    try
    {
        throw customException::UserInputExceptionWarning(
            std::format("{}Density and box dimensions set. Density will be ignored.", _OUTPUT_));
    }
    catch (const customException::UserInputExceptionWarning &e)
    {
        std::cout << _OUTPUT_ << e.what() << '\n' << '\n' << std::flush;
    }
}

/**
 * @brief write a message to the stdout to inform about the setup
 *
 * @param momentum
 */
void StdoutOutput::writeSetup(const std::string &setup) const { std::cout << setupMessage(setup) << '\n' << std::flush; }

/**
 * @brief write a message to the stdout to inform that the setup is completed
 *
 * @param momentum
 */
void StdoutOutput::writeSetupCompleted() const { std::cout << setupCompletedMessage() << '\n' << std::flush; }

/**
 * @brief write a message to the stdout to inform about the start of reading a file
 *
 */
void StdoutOutput::writeRead(const std::string &file) const { std::cout << readMessage(file) << '\n' << std::flush; }