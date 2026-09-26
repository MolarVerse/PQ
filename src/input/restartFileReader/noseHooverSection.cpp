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

#include "noseHooverSection.hpp"

#include <format>   // for format
#include <string>   // for string

#include "exceptions.hpp"           // for exc::RstFileException
#include "thermostatSettings.hpp"   // for ThermostatSettings

namespace input::restartFile
{

    /**
     * @brief checks the number of arguments in the line
     *
     * @param lineElements all elements of the line
     *
     * @throws exc::RstFileException if the number of arguments is not
     * correct
     */
    void NoseHooverSection::process(
        std::vector<std::string> &lineElements,
        engine::Engine & /*engine*/
    )
    {
        if (4 != lineElements.size())
        {
            throw exc::RstFileException(
                std::format(
                    "Error not enough arguments in line {} for a chi entry of "
                    "the "
                    "nose "
                    "hoover thermostat",
                    _lineNumber
                )
            );
        }

        const auto idx  = stoul(lineElements[1]);
        const auto chi  = stod(lineElements[2]);
        const auto zeta = stod(lineElements[3]);

        auto [iterChi, chiIsInserted] =
            settings::ThermostatSettings::addChi(idx, chi);
        auto [iterZeta, zetaIsInserted] =
            settings::ThermostatSettings::addZeta(idx, zeta);

        if (!chiIsInserted || !zetaIsInserted)
        {
            throw exc::RstFileException(
                std::format(
                    "Error in line {} in restart file; chi or zeta entry "
                    "already "
                    "exists",
                    _lineNumber
                )
            );
        }
    }

    /**
     * @brief returns the keyword of the section
     *
     * @return std::string "chi"
     */
    std::string NoseHooverSection::keyword() { return "chi"; }

    /**
     * @brief returns if the section is a header
     *
     * @return bool true
     */
    bool NoseHooverSection::isHeader() { return true; }

}   // namespace input::restartFile
