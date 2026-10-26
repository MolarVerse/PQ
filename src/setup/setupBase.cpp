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

#include "setupBase.hpp"

#include "stdoutOutput.hpp"

namespace setup
{

    /**
     * @brief Writes a setup warning to the log and stdout
     *
     * @param log The log output object
     * @param message The warning message
     */
    void SetupBase::writeSetupWarning(
        out::LogOutput    &log,
        const std::string &message
    )
    {
        log.writeSetupWarning(message);
        out::StdoutOutput::writeSetupWarning(message);
    }

}   // namespace setup
