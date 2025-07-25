/*****************************************************************************
<GPL_HEADER>

    PQ
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

#include "logOutput.hpp"

#include <format>    // for format
#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <string>    // for char_traits, operator<<

#include "outputMessages.hpp"   // for initialMomentumMessage

using output::LogOutput;

/**
 * @brief write an empty line to the log file
 *
 */
void LogOutput::writeEmptyLine() { _fp << '\n' << std::flush; }

/**
 * @brief write a message to the log file
 *
 * @param message
 */
void LogOutput::writeInfo(const std::string &message)
{
    _fp << message << '\n' << std::flush;
}

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
 * @brief write a warning message to the log file if density and box dimensions
 * are set
 *
 */
void LogOutput::writeDensityWarning()
{
    _fp << _WARNING_
        << "Density and box dimensions set. Density will be ignored."
        << "\n\n"
        << std::flush;
}

/**
 * @brief write a warning message to the log file if the optimization did not
 * converge
 *
 * @param message
 */
void LogOutput::writeOptWarning(const std::string &message)
{
    _fp << _WARNING_ << message << "\n\n" << std::flush;
}

/**
 * @brief write initial momentum to log file
 *
 * @param momentum
 */
void LogOutput::writeInitialMomentum(const double momentum)
{
    _fp << "\n" << std::flush;

    _fp << std::format(
        "{}Initial momentum = {:.5e} {}*amu/fs\n",
        _INFO_,
        momentum,
        _ANGSTROM_
    );

    _fp << std::flush;
}

/**
 * @brief write a message to inform about the start of the setup
 *
 */
void LogOutput::writeSetup(const std::string &setup)
{
    _fp << setupMessage(setup) << '\n' << std::flush;
}

/**
 * @brief write a message to inform about the setup
 *
 */
void LogOutput::writeSetupInfo(const std::string &setupInfo)
{
    _fp << _OUTPUT_ << setupInfo << '\n' << std::flush;
}

/**
 * @brief write a message to issue a warning about the setup
 *
 */
void LogOutput::writeSetupWarning(const std::string &setupWarning)
{
    _fp << _WARNING_ << setupWarning << '\n' << std::flush;
}

/**
 * @brief write a message to the stdout to inform that the setup is completed
 *
 * @param momentum
 */
void LogOutput::writeSetupCompleted()
{
    _fp << setupCompletedMessage() << '\n' << std::flush;
}

/**
 * @brief write a message to inform about starting to read a file
 *
 */
void LogOutput::writeRead(const std::string &message, const std::string &file)
{
    _fp << readMessage(message, file) << '\n' << std::flush;
}