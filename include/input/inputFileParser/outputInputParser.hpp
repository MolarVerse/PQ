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

#ifndef _OUTPUT_INPUT_PARSER_HPP_

#define _OUTPUT_INPUT_PARSER_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

namespace input
{
    /**
     * @brief OutputInputParser inherits from InputFileParser
     *
     * @details Parses the output commands in the input file
     *
     */
    class OutputInputParser : public InputFileParser
    {
       public:
        OutputInputParser();

        void addOverwriteOutputKeyword();
        void addIncludeOutputMetadataKeyword();
        void addOutputFrequencyKeyword();
        void addFilePrefixKeyword();

        void addLogFilenameKeyword();
        void addReferenceFilenameKeyword();
        void addInfoFilenameKeyword();
        void addEnergyFilenameKeyword();
        void addInstantEnergyFilenameKeyword();
        void addTrajectoryFilenameKeyword();
        void addHybridCenterFilenameKeyword();
        void addVelocityFilenameKeyword();
        void addForceFilenameKeyword();
        void addRestartFilenameKeyword();
        void addChargeFilenameKeyword();
        void addMomentumFilenameKeyword();
        void addVirialFilenameKeyword();
        void addStressFilenameKeyword();
        void addBoxFilenameKeyword();
        void addTimingsFilenameKeyword();
        void addOptFilenameKeyword();

        void addRPMDRestartFilenameKeyword();
        void addRPMDTrajectoryFilenameKeyword();
        void addRPMDVelocityFilenameKeyword();
        void addRPMDForceFilenameKeyword();
        void addRPMDChargeFilenameKeyword();
        void addRPMDEnergyFilenameKeyword();
    };

}   // namespace input

#endif   // _OUTPUT_INPUT_PARSER_HPP_
