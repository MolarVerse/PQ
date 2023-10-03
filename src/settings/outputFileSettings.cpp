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

#include "outputFileSettings.hpp"

#include <cstdint>   // for UINT64_MAX

using settings::OutputFileSettings;

/**
 * @brief Sets the output frequency of the simulation
 *
 * @param outputFreq
 *
 * @throw InputFileException if output frequency is negative
 *
 * @note
 *  if output frequency is 0, it is set to UINT64_MAX
 *  in order to avoid division by 0 in the output
 *
 */
void OutputFileSettings::setOutputFrequency(const size_t outputFreq)
{
    if (0 == outputFreq)
        _outputFrequency = UINT64_MAX;
    else
        _outputFrequency = outputFreq;
}

/**
 * @brief returns the reference file name
 *
 * @details in order to avoid overwriting the reference file and not to set it in the input file - the reference file name is set
 * to the log file name + ".ref"
 *
 * @param restartFileName
 */
std::string OutputFileSettings::getReferenceFileName() { return _logFileName + ".ref"; }