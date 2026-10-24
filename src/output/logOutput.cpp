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

#include <format>
#include <ostream>
#include <string>

#include "outputMessages.hpp"

namespace out
{

    /**
     * @brief write an empty line to the log file
     *
     */
    void LogOutput::writeEmptyLine()
    {
        _getFilePointer() << '\n' << std::flush;
    }

    /**
     * @brief write a message to the log file
     *
     * @param message
     */
    void LogOutput::writeInfo(const std::string &message)
    {
        _getFilePointer() << message << '\n' << std::flush;
    }

    /**
     * @brief write header title
     *
     */
    void LogOutput::writeHeader()
    {
        _getFilePointer() << header() << '\n' << std::flush;
    }

    /**
     * @brief write a message to the log file if the simulation ended normally
     *
     */
    void LogOutput::writeEndedNormally(double elapsedTime)
    {
        _getFilePointer() << elapsedTimeMessage(elapsedTime) << '\n';
        _getFilePointer() << endedNormally() << '\n' << std::flush;
    }

    /**
     * @brief write a warning message to the log file if density and box
     * dimensions are set
     *
     */
    void LogOutput::writeDensityWarning()
    {
        _getFilePointer(
        ) << WARNING
          << "Density and box dimensions set. Density will be ignored."
          << "\n\n"
          << std::flush;
    }

    /**
     * @brief write a warning message to the log file if the optimization did
     * not converge
     *
     * @param message
     */
    void LogOutput::writeOptWarning(const std::string &message)
    {
        _getFilePointer() << WARNING << message << "\n\n" << std::flush;
    }

    /**
     * @brief write initial momentum to log file
     *
     * @param momentum
     */
    void LogOutput::writeInitialMomentum(double momentum)
    {
        _getFilePointer() << "\n" << std::flush;

        _getFilePointer() << std::format(
            "{}Initial momentum = {:.5e} {}*amu/fs\n",
            INFO,
            momentum,
            ANGSTROM
        );

        _getFilePointer() << std::flush;
    }

    /**
     * @brief write a message to inform about the start of the setup
     *
     */
    void LogOutput::writeSetup(const std::string &setup)
    {
        _getFilePointer() << setupMessage(setup) << '\n' << std::flush;
    }

    /**
     * @brief write a message to inform about the setup
     *
     */
    void LogOutput::writeSetupInfo(const std::string &setupInfo)
    {
        _getFilePointer() << OUTPUT << setupInfo << '\n' << std::flush;
    }

    /**
     * @brief write a message to issue a warning about the setup
     *
     */
    void LogOutput::writeSetupWarning(const std::string &setupWarning)
    {
        _getFilePointer() << WARNING << setupWarning << '\n' << std::flush;
    }

    /**
     * @brief write a message to the stdout to inform that the setup is
     * completed
     *
     */
    void LogOutput::writeSetupCompleted()
    {
        _getFilePointer() << setupCompletedMessage() << '\n' << std::flush;
    }

    /**
     * @brief write a message to inform about starting to read a file
     *
     */
    void LogOutput::writeRead(
        const std::string &message,
        const std::string &file
    )
    {
        _getFilePointer() << readMessage(message, file) << '\n' << std::flush;
    }

    /**
     * @brief add a warning message that will be printed after the output file
     * has been created
     *
     */
    void LogOutput::queueWarning(const std::string &warning)
    {
        _pendingWarnings.push_back(warning);
    }

    /**
     * @brief flush pending warning messages to the output file
     *
     */
    void LogOutput::flushQueuedWarnings()
    {
        for (const auto &message : _pendingWarnings)
        {
            writeSetupWarning(message);
            writeEmptyLine();
        }

        _pendingWarnings.clear();
    }

}   // namespace out
