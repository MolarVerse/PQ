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