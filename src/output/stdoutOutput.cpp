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

#include "stdoutOutput.hpp"

#include <format>     // for format
#include <iostream>   // for operator<<, char_traits, basic_ostream, cout
#include <string>     // for operator<<

#include "exceptions.hpp"   // for UserInputExceptionWarning, customException
#include "outputMessages.hpp"   // for initialMomentumMessage

namespace out
{

    /**
     * @brief write a message to the stdout
     *
     * @param message
     */
    void StdoutOutput::writeInfo(const std::string &message) const
    {
        std::cout << message << '\n' << std::flush;
    }

    /**
     * @brief write header title
     *
     * @return string
     */
    void StdoutOutput::writeHeader() const
    {
        std::cout << header() << '\n' << std::flush;
    }

    /**
     * @brief write a message to the stdout if the simulation ended normally
     *
     * @param elapsedTime
     */
    void StdoutOutput::writeEndedNormally(const double elapsedTime) const
    {
        std::cout << elapsedTimeMessage(elapsedTime) << '\n';
        std::cout << endedNormally() << '\n' << std::flush;
    }

    /**
     * @brief write a warning message to the stdout if density and box
     * dimensions are set
     *
     */
    void StdoutOutput::writeDensityWarning() const
    {
        try
        {
            throw exc::UserInputExceptionWarning(
                std::format(
                    "{}Density and box dimensions set. Density will be "
                    "ignored.",
                    OUTPUT
                )
            );
        }
        catch (const exc::UserInputExceptionWarning &e)
        {
            std::cout << OUTPUT << e.what() << "\n\n" << std::flush;
        }
    }

    /**
     * @brief write a warning message to the stdout if the optimization did not
     * converge
     *
     * @param msg
     */
    void StdoutOutput::writeOptWarning(const std::string &msg) const
    {
        try
        {
            throw exc::UserInputExceptionWarning(
                std::format("{}{}", WARNING, msg)
            );
        }
        catch (const exc::UserInputExceptionWarning &e)
        {
            std::cout << e.what() << "\n\n" << std::flush;
        }
    }

    /**
     * @brief write a message to the stdout to inform about the setup
     *
     * @param setup
     */
    void StdoutOutput::writeSetup(const std::string &setup) const
    {
        std::cout << setupMessage(setup) << '\n' << std::flush;
    }

    /**
     * @brief write a message to the stdout to issue a warning about the setup
     *
     * @param warning
     */
    void StdoutOutput::writeSetupWarning(const std::string &warning) const
    {
        std::cout << WARNING << (warning) << "\n\n" << std::flush;
    }

    /**
     * @brief write a message to the stdout to inform that the setup is
     * completed
     *
     */
    void StdoutOutput::writeSetupCompleted() const
    {
        std::cout << setupCompletedMessage() << '\n' << std::flush;
    }

    /**
     * @brief write a message to the stdout to inform about the start of reading
     * a file
     *
     * @param message, file
     */
    void StdoutOutput::writeRead(
        const std::string &message,
        const std::string &file
    ) const
    {
        std::cout << readMessage(message, file) << '\n' << std::flush;
    }
}   // namespace out
