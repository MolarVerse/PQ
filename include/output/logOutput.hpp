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

#ifndef _LOG_OUTPUT_HPP_

#define _LOG_OUTPUT_HPP_

#include "output.hpp"

namespace output
{
    /**
     * @class LogOutput inherits from Output
     *
     * @brief Output file for log file
     *
     */
    class LogOutput : public Output
    {
      public:
        using Output::Output;

        void writeEmptyLine() { _fp << '\n' << std::flush; }

        void writeHeader();
        void writeEndedNormally(const double elapsedTime);

        void writeDensityWarning();
        void writeInitialMomentum(const double momentum);

        void writeSetup(const std::string &setup);
        void writeSetupInfo(const std::string &setupInfo);
        void writeSetupCompleted();
        void writeRead(const std::string &file);
    };

}   // namespace output

#endif   // _LOG_OUTPUT_HPP_