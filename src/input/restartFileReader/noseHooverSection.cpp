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

#include "noseHooverSection.hpp"

#include "exceptions.hpp"           // for RstFileException
#include "thermostatSettings.hpp"   // for ThermostatSettings

#include <format>   // for format
#include <string>   // for string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

using input::restartFile::NoseHooverSection;

/**
 * @brief checks the number of arguments in the line
 *
 * @param lineElements all elements of the line
 *
 * @throws customException::RstFileException if the number of arguments is not correct
 */
void NoseHooverSection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (4 != lineElements.size())
        throw customException::RstFileException(
            std::format("Error not enough arguments in line {} for a chi entry of the nose hoover thermostat", _lineNumber));

    auto [iterChi, chiIsInserted]   = settings::ThermostatSettings::addChi(stoul(lineElements[1]), stod(lineElements[2]));
    auto [iterZeta, zetaIsInserted] = settings::ThermostatSettings::addZeta(stoul(lineElements[1]), stod(lineElements[3]));

    if (!chiIsInserted || !zetaIsInserted)
        throw customException::RstFileException(
            std::format("Error in line {} in restart file; chi or zeta entry already exists", _lineNumber));
}