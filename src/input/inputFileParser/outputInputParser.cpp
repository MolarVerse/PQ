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

#include "outputInputParser.hpp"

#include <format>       // for format
#include <functional>   // for bind_front, _Bind_front_t

#include "exceptions.hpp"           // for InputFileException
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "stringUtilities.hpp"      // for toLowerCopy

using namespace input;
using namespace engine;
using namespace utilities;
using namespace customException;
using namespace settings;

/**
 * @brief Construct a new Input File Parser Output:: Input File Parser Output
 * object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap:
 * 1)  output_freq <size_t>
 * 2)  file_prefix <string>
 * 3)  output_file <string>
 * 4)  ref_file <string>
 * 5)  info_file <string>
 * 6)  energy_file <string>
 * 7)  instant_energy_file <string>
 * 8)  traj_file <string>
 * 9)  vel_file <string>
 * 10) force_file <string>
 * 11) restart_file <string>
 * 12) charge_file <string>
 * 13) momentum_file <string>
 * 14) virial_file <string>
 * 15) stress_file <string>
 * 16) box_file <string>
 * 17) timings_file <string>
 * 18) opt_file <string>
 * 19) rpmd_restart_file <string>
 * 20) rpmd_traj_file <string>
 * 21) rpmd_vel_file <string>
 * 22) rpmd_force_file <string>
 * 23) rpmd_charge_file <string>
 * 24) rpmd_energy_file <string>
 *
 * @param engine
 */
OutputInputParser::OutputInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("output_freq"),
        bind_front(&OutputInputParser::parseOutputFreq, this),
        false
    );
    addKeyword(
        std::string("file_prefix"),
        bind_front(&OutputInputParser::parseFilePrefix, this),
        false
    );
    addKeyword(
        std::string("output_file"),
        bind_front(&OutputInputParser::parseLogFilename, this),
        false
    );
    addKeyword(
        std::string("reference_file"),
        bind_front(&OutputInputParser::parseRefFilename, this),
        false
    );
    addKeyword(
        std::string("info_file"),
        bind_front(&OutputInputParser::parseInfoFilename, this),
        false
    );
    addKeyword(
        std::string("energy_file"),
        bind_front(&OutputInputParser::parseEnergyFilename, this),
        false
    );
    addKeyword(
        std::string("instant_energy_file"),
        bind_front(&OutputInputParser::parseInstantEnergyFilename, this),
        false
    );
    addKeyword(
        std::string("traj_file"),
        bind_front(&OutputInputParser::parseTrajectoryFilename, this),
        false
    );
    addKeyword(
        std::string("vel_file"),
        bind_front(&OutputInputParser::parseVelocityFilename, this),
        false
    );
    addKeyword(
        std::string("force_file"),
        bind_front(&OutputInputParser::parseForceFilename, this),
        false
    );
    addKeyword(
        std::string("restart_file"),
        bind_front(&OutputInputParser::parseRestartFilename, this),
        false
    );
    addKeyword(
        std::string("charge_file"),
        bind_front(&OutputInputParser::parseChargeFilename, this),
        false
    );
    addKeyword(
        std::string("momentum_file"),
        bind_front(&OutputInputParser::parseMomentumFilename, this),
        false
    );
    addKeyword(
        std::string("virial_file"),
        bind_front(&OutputInputParser::parseVirialFilename, this),
        false
    );
    addKeyword(
        std::string("stress_file"),
        bind_front(&OutputInputParser::parseStressFilename, this),
        false
    );
    addKeyword(
        std::string("box_file"),
        bind_front(&OutputInputParser::parseBoxFilename, this),
        false
    );
    addKeyword(
        std::string("timings_file"),
        bind_front(&OutputInputParser::parseTimingsFilename, this),
        false
    );
    addKeyword(
        std::string("opt_file"),
        bind_front(&OutputInputParser::parseOptFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_restart_file"),
        bind_front(&OutputInputParser::parseRPMDRestartFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_traj_file"),
        bind_front(&OutputInputParser::parseRPMDTrajectoryFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_vel_file"),
        bind_front(&OutputInputParser::parseRPMDVelocityFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_force_file"),
        bind_front(&OutputInputParser::parseRPMDForceFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_charge_file"),
        bind_front(&OutputInputParser::parseRPMDChargeFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_energy_file"),
        bind_front(&OutputInputParser::parseRPMDEnergyFilename, this),
        false
    );
    addKeyword(
        std::string("overwrite_output"),
        bind_front(&OutputInputParser::parseOverwriteOutput, this),
        false
    );
}

/**
 * @brief parse output frequency of simulation and set it in output statically
 *
 * @details default value is 1
 *
 * @param lineElements
 *
 * @throws InputFileException if output frequency is negative
 */
void OutputInputParser::parseOutputFreq(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto outputFrequency = stoi(lineElements[2]);
    if (outputFrequency < 0)
        throw InputFileException(format(
            "Output frequency cannot be negative - \"{}\" at line {} in input "
            "file",
            lineElements[2],
            lineNumber
        ));

    OutputFileSettings::setOutputFrequency(size_t(outputFrequency));
}

/**
 * @brief parse file prefix of simulation and set it in output statically
 *
 * @details default value is default
 *
 * @param lineElements
 */
void OutputInputParser::parseFilePrefix(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setFilePrefix(lineElements[2]);
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @details default value is default.log
 *
 * @param lineElements
 */
void OutputInputParser::parseLogFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setLogFileName(lineElements[2]);
}

/**
 * @brief parse ref filename of simulation and add it to output
 *
 * @details default value is default.ref
 *
 * @param lineElements
 */
void OutputInputParser::parseRefFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRefFileName(lineElements[2]);
}

/**
 * @brief parse info filename of simulation and add it to output
 *
 * @details default value is default.info
 *
 * @param lineElements
 */
void OutputInputParser::parseInfoFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setInfoFileName(lineElements[2]);
}

/**
 * @brief parse energy filename of simulation and add it to output
 *
 * @details default value is default.en
 *
 * @param lineElements
 */
void OutputInputParser::parseEnergyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setEnergyFileName(lineElements[2]);
}

/**
 * @brief parse instant energy filename of simulation and add it to output
 *
 * @details default value is default.inen
 *
 * @param lineElements
 */
void OutputInputParser::parseInstantEnergyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setInstantEnergyFileName(lineElements[2]);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @details default value is default.xyz
 *
 * @param lineElements
 */
void OutputInputParser::parseTrajectoryFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setTrajectoryFileName(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @details default value is default.vel
 *
 * @param lineElements
 */
void OutputInputParser::parseVelocityFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setVelocityFileName(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @details default value is default.force
 *
 * @param lineElements
 */
void OutputInputParser::parseForceFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setForceFileName(lineElements[2]);
}

/**
 * @brief parse restart filename of simulation and add it to output
 *
 * @details default value is default.rst
 *
 * @param lineElements
 */
void OutputInputParser::parseRestartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRestartFileName(lineElements[2]);
}

/**
 * @brief parse charge filename of simulation and add it to output
 *
 * @details default value is default.chrg
 *
 * @param lineElements
 */
void OutputInputParser::parseChargeFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setChargeFileName(lineElements[2]);
}

/**
 * @brief parse momentum filename of simulation and add it to output
 *
 * @details default value is default.mom
 *
 * @param lineElements
 */
void OutputInputParser::parseMomentumFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setMomentumFileName(lineElements[2]);
}

/**
 * @brief parse virial filename of simulation and add it to output
 *
 * @details default value is default.vir
 *
 * @param lineElements
 */
void OutputInputParser::parseVirialFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setVirialFileName(lineElements[2]);
}

/**
 * @brief parse stress filename of simulation and add it to output
 *
 * @details default value is default.stress
 *
 * @param lineElements
 */
void OutputInputParser::parseStressFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setStressFileName(lineElements[2]);
}

/**
 * @brief parse box filename of simulation and add it to output
 *
 * @details default value is default.box
 *
 * @param lineElements
 */
void OutputInputParser::parseBoxFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setBoxFileName(lineElements[2]);
}

/**
 * @brief parse timings filename of simulation and add it to output
 *
 * @details default value is default.timings
 *
 * @param lineElements
 */
void OutputInputParser::parseTimingsFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setTimingsFileName(lineElements[2]);
}

/**
 * @brief parse optimization filename of simulation and add it to output
 *
 * @details default value is default.opt
 *
 * @param lineElements
 */
void OutputInputParser::parseOptFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setOptFileName(lineElements[2]);
}

/**
 * @brief parse RPMD restart filename of simulation and add it to output
 *
 * @details default value is default.rpmd.rst
 *
 * @param lineElements
 */
void OutputInputParser::parseRPMDRestartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRingPolymerRestartFileName(lineElements[2]);
}

/**
 * @brief parse RPMD trajectory filename of simulation and add it to output
 *
 * @details default value is default.rpmd.xyz
 *
 * @param lineElements
 */
void OutputInputParser::parseRPMDTrajectoryFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRingPolymerTrajectoryFileName(lineElements[2]);
}

/**
 * @brief parse RPMD velocity filename of simulation and add it to output
 *
 * @details default value is default.rpmd.vel
 *
 * @param lineElements
 */
void OutputInputParser::parseRPMDVelocityFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRingPolymerVelocityFileName(lineElements[2]);
}

/**
 * @brief parse RPMD force filename of simulation and add it to output
 *
 * @details default value is default.rpmd.force
 *
 * @param lineElements
 */
void OutputInputParser::parseRPMDForceFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRingPolymerForceFileName(lineElements[2]);
}

/**
 * @brief parse RPMD charge filename of simulation and add it to output
 *
 * @details default value is default.rpmd.chrg
 *
 * @param lineElements
 */
void OutputInputParser::parseRPMDChargeFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRingPolymerChargeFileName(lineElements[2]);
}

/**
 * @brief parse RPMD energy filename of simulation and add it to output
 *
 * @details default value is default.rpmd.en
 *
 * @param lineElements
 */
void OutputInputParser::parseRPMDEnergyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    OutputFileSettings::setRingPolymerEnergyFileName(lineElements[2]);
}

/**
 * @brief parse if existing output files should be overwritten
 *
 * @param lineElements
 * @param lineNumber
 */
void OutputInputParser::parseOverwriteOutput(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    OutputFileSettings::setOverwriteOutputFiles(keywordToBool(lineElements));
}