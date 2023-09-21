#include "inputFileParserOutput.hpp"

#include "energyOutput.hpp"         // for EnergyOutput
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for InputFileException
#include "infoOutput.hpp"           // for InfoOutput
#include "logOutput.hpp"            // for LogOutput
#include "output.hpp"               // for Output, output
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "rstFileOutput.hpp"        // for RstFileOutput
#include "trajectoryOutput.hpp"     // for TrajectoryOutput

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Output:: Input File Parser Output object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) output_freq <size_t>
 * 2) output_file <string>
 * 3) info_file <string>
 * 4) energy_file <string>
 * 5) traj_file <string>
 * 6) vel_file <string>
 * 7) force_file <string>
 * 8) restart_file <string>
 * 9) charge_file <string>
 *
 * @param engine
 */
InputFileParserOutput::InputFileParserOutput(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("output_freq"), bind_front(&InputFileParserOutput::parseOutputFreq, this), false);
    addKeyword(std::string("output_file"), bind_front(&InputFileParserOutput::parseLogFilename, this), false);
    addKeyword(std::string("info_file"), bind_front(&InputFileParserOutput::parseInfoFilename, this), false);
    addKeyword(std::string("energy_file"), bind_front(&InputFileParserOutput::parseEnergyFilename, this), false);
    addKeyword(std::string("traj_file"), bind_front(&InputFileParserOutput::parseTrajectoryFilename, this), false);
    addKeyword(std::string("vel_file"), bind_front(&InputFileParserOutput::parseVelocityFilename, this), false);
    addKeyword(std::string("force_file"), bind_front(&InputFileParserOutput::parseForceFilename, this), false);
    addKeyword(std::string("restart_file"), bind_front(&InputFileParserOutput::parseRestartFilename, this), false);
    addKeyword(std::string("charge_file"), bind_front(&InputFileParserOutput::parseChargeFilename, this), false);
    addKeyword(std::string("rpmd_restart_file"), bind_front(&InputFileParserOutput::parseRPMDRestartFilename, this), false);
    addKeyword(std::string("rpmd_traj_file"), bind_front(&InputFileParserOutput::parseRPMDTrajectoryFilename, this), false);
    addKeyword(std::string("rpmd_vel_file"), bind_front(&InputFileParserOutput::parseRPMDVelocityFilename, this), false);
    addKeyword(std::string("rpmd_force_file"), bind_front(&InputFileParserOutput::parseRPMDForceFilename, this), false);
    addKeyword(std::string("rpmd_charge_file"), bind_front(&InputFileParserOutput::parseRPMDChargeFilename, this), false);
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
void InputFileParserOutput::parseOutputFreq(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto outputFrequency = stoi(lineElements[2]);
    if (outputFrequency < 0)
        throw customException::InputFileException(
            format("Output frequency cannot be negative - \"{}\" at line {} in input file", lineElements[2], lineNumber));

    settings::OutputFileSettings::setOutputFrequency(size_t(outputFrequency));
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @details default value is default.out
 *
 * @param lineElements
 */
void InputFileParserOutput::parseLogFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
void InputFileParserOutput::parseInfoFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
void InputFileParserOutput::parseEnergyFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setEnergyFileName(lineElements[2]);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @details default value is default.xyz
 *
 * @param lineElements
 */
void InputFileParserOutput::parseTrajectoryFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
void InputFileParserOutput::parseVelocityFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
void InputFileParserOutput::parseForceFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
void InputFileParserOutput::parseRestartFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
void InputFileParserOutput::parseChargeFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setChargeFileName(lineElements[2]);
}

/**
 * @brief parse RPMD restart filename of simulation and add it to output
 *
 * @details default value is default.rpmd.rst
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDRestartFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerRestartFileName(lineElements[2]);
}

/**
 * @brief parse RPMD trajectory filename of simulation and add it to output
 *
 * @details default value is default.rpmd.xyz
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDTrajectoryFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerTrajectoryFileName(lineElements[2]);
}

/**
 * @brief parse RPMD velocity filename of simulation and add it to output
 *
 * @details default value is default.rpmd.vel
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDVelocityFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerVelocityFileName(lineElements[2]);
}

/**
 * @brief parse RPMD force filename of simulation and add it to output
 *
 * @details default value is default.rpmd.force
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRPMDForceFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
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
void InputFileParserOutput::parseRPMDChargeFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    settings::OutputFileSettings::setRingPolymerChargeFileName(lineElements[2]);
}