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

#include "inputFileParserOutput.hpp"

#include <format>       // for format
#include <functional>   // for bind_front, _Bind_front_t

#include "exceptions.hpp"           // for InputFileException
#include "outputFileSettings.hpp"   // for OutputFileSettings

using namespace input;

/**
 * @brief Construct a new Input File Parser Output:: Input File Parser Output
 * object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) output_freq <size_t> 2)
 * output_file <string> 3) info_file <string> 4) energy_file <string> 5)
 * instant_energy_file <string> 6) traj_file <string> 7) vel_file <string> 8)
 * force_file <string> 9) restart_file <string> 10) charge_file <string> 11)
 * momentum_file <string> 12) virial_file <string> 13) stress_file <string> 14)
 * box_file <string> 15) rpmd_restart_file <string> 16) rpmd_traj_file <string>
 * 17) rpmd_vel_file <string>
 * 18) rpmd_force_file <string>
 * 19) rpmd_charge_file <string>
 * 20) rpmd_energy_file <string>
 *
 * @param engine
 */
InputFileParserOutput::InputFileParserOutput(engine::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("output_freq"),
        bind_front(&InputFileParserOutput::parseOutputFreq, this),
        false
    );
    addKeyword(
        std::string("file_prefix"),
        bind_front(&InputFileParserOutput::parseFilePrefix, this),
        false
    );

    addKeyword(
        std::string("output_file"),
        bind_front(&InputFileParserOutput::parseLogFilename, this),
        false
    );
    addKeyword(
        std::string("info_file"),
        bind_front(&InputFileParserOutput::parseInfoFilename, this),
        false
    );
    addKeyword(
        std::string("energy_file"),
        bind_front(&InputFileParserOutput::parseEnergyFilename, this),
        false
    );
    addKeyword(
        std::string("instant_energy_file"),
        bind_front(&InputFileParserOutput::parseInstantEnergyFilename, this),
        false
    );
    addKeyword(
        std::string("traj_file"),
        bind_front(&InputFileParserOutput::parseTrajectoryFilename, this),
        false
    );
    addKeyword(
        std::string("vel_file"),
        bind_front(&InputFileParserOutput::parseVelocityFilename, this),
        false
    );
    addKeyword(
        std::string("force_file"),
        bind_front(&InputFileParserOutput::parseForceFilename, this),
        false
    );
    addKeyword(
        std::string("restart_file"),
        bind_front(&InputFileParserOutput::parseRestartFilename, this),
        false
    );
    addKeyword(
        std::string("charge_file"),
        bind_front(&InputFileParserOutput::parseChargeFilename, this),
        false
    );
    addKeyword(
        std::string("momentum_file"),
        bind_front(&InputFileParserOutput::parseMomentumFilename, this),
        false
    );

    addKeyword(
        std::string("virial_file"),
        bind_front(&InputFileParserOutput::parseVirialFilename, this),
        false
    );
    addKeyword(
        std::string("stress_file"),
        bind_front(&InputFileParserOutput::parseStressFilename, this),
        false
    );
    addKeyword(
        std::string("box_file"),
        bind_front(&InputFileParserOutput::parseBoxFilename, this),
        false
    );
    addKeyword(
        std::string("opt_file"),
        bind_front(&InputFileParserOutput::parseOptFilename, this),
        false
    );

    addKeyword(
        std::string("rpmd_restart_file"),
        bind_front(&InputFileParserOutput::parseRPMDRestartFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_traj_file"),
        bind_front(&InputFileParserOutput::parseRPMDTrajectoryFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_vel_file"),
        bind_front(&InputFileParserOutput::parseRPMDVelocityFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_force_file"),
        bind_front(&InputFileParserOutput::parseRPMDForceFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_charge_file"),
        bind_front(&InputFileParserOutput::parseRPMDChargeFilename, this),
        false
    );
    addKeyword(
        std::string("rpmd_energy_file"),
        bind_front(&InputFileParserOutput::parseRPMDEnergyFilename, this),
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
void InputFileParserOutput::parseOutputFreq(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto outputFrequency = stoi(lineElements[2]);
    if (outputFrequency < 0)
        throw customException::InputFileException(format(
            "Output frequency cannot be negative - \"{}\" at line {} in input "
            "file",
            lineElements[2],
            lineNumber
        ));

    settings::OutputFileSettings::setOutputFrequency(size_t(outputFrequency));
}

/**
 * @brief parse file prefix of simulation and set it in output statically
 *
 * @details default value is default
 *
 * @param lineElements
 */
void InputFileParserOutput::parseFilePrefix(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setFilePrefix(lineElements[2]);
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @details default value is default.out
 *
 * @param lineElements
 */
void InputFileParserOutput::parseLogFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setLogFileName(lineElements[2]);
}

/**
 * @brief parse info filename of simulation and add it to output
 *
 * @details default value is default.info
 *
 * @param lineElements
 */
void InputFileParserOutput::parseInfoFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setInfoFileName(lineElements[2]);
}

/**
 * @brief parse energy filename of simulation and add it to output
 *
 * @details default value is default.en
 *
 * @param lineElements
 */
void InputFileParserOutput::parseEnergyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setEnergyFileName(lineElements[2]);
}

/**
 * @brief parse instant energy filename of simulation and add it to output
 *
 * @details default value is default.inen
 *
 * @param lineElements
 */
void InputFileParserOutput::parseInstantEnergyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setInstantEnergyFileName(lineElements[2]);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @details default value is default.xyz
 *
 * @param lineElements
 */
void InputFileParserOutput::parseTrajectoryFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setTrajectoryFileName(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @details default value is default.vel
 *
 * @param lineElements
 */
void InputFileParserOutput::parseVelocityFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setVelocityFileName(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @details default value is default.force
 *
 * @param lineElements
 */
void InputFileParserOutput::parseForceFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setForceFileName(lineElements[2]);
}

/**
 * @brief parse restart filename of simulation and add it to output
 *
 * @details default value is default.rst
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRestartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRestartFileName(lineElements[2]);
}

/**
 * @brief parse charge filename of simulation and add it to output
 *
 * @details default value is default.chrg
 *
 * @param lineElements
 */
void InputFileParserOutput::parseChargeFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setChargeFileName(lineElements[2]);
}

/**
 * @brief parse momentum filename of simulation and add it to output
 *
 * @details default value is default.mom
 *
 * @param lineElements
 */
void InputFileParserOutput::parseMomentumFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setMomentumFileName(lineElements[2]);
}

/**
 * @brief parse virial filename of simulation and add it to output
 *
 * @details default value is default.vir
 *
 * @param lineElements
 */
void InputFileParserOutput::parseVirialFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setVirialFileName(lineElements[2]);
}

/**
 * @brief parse stress filename of simulation and add it to output
 *
 * @details default value is default.stress
 *
 * @param lineElements
 */
void InputFileParserOutput::parseStressFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setStressFileName(lineElements[2]);
}

/**
 * @brief parse box filename of simulation and add it to output
 *
 * @details default value is default.box
 *
 * @param lineElements
 */
void InputFileParserOutput::parseBoxFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setBoxFileName(lineElements[2]);
}

/**
 * @brief parse optimization filename of simulation and add it to output
 *
 * @details default value is default.opt
 *
 * @param lineElements
 */
void InputFileParserOutput::parseOptFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setOptFileName(lineElements[2]);
}

/**
 * @brief parse RPMD restart filename of simulation and add it to output
 *
 * @details default value is default.rpmd.rst
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDRestartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerRestartFileName(lineElements[2]
    );
}

/**
 * @brief parse RPMD trajectory filename of simulation and add it to output
 *
 * @details default value is default.rpmd.xyz
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDTrajectoryFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerTrajectoryFileName(
        lineElements[2]
    );
}

/**
 * @brief parse RPMD velocity filename of simulation and add it to output
 *
 * @details default value is default.rpmd.vel
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDVelocityFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerVelocityFileName(lineElements[2]
    );
}

/**
 * @brief parse RPMD force filename of simulation and add it to output
 *
 * @details default value is default.rpmd.force
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDForceFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerForceFileName(lineElements[2]);
}

/**
 * @brief parse RPMD charge filename of simulation and add it to output
 *
 * @details default value is default.rpmd.chrg
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDChargeFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerChargeFileName(lineElements[2]);
}

/**
 * @brief parse RPMD energy filename of simulation and add it to output
 *
 * @details default value is default.rpmd.en
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDEnergyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerEnergyFileName(lineElements[2]);
}