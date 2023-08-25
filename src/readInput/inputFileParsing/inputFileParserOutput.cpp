#include "inputFileParserOutput.hpp"

#include "energyOutput.hpp"       // for EnergyOutput
#include "engine.hpp"             // for Engine
#include "exceptions.hpp"         // for InputFileException
#include "infoOutput.hpp"         // for InfoOutput
#include "logOutput.hpp"          // for LogOutput
#include "output.hpp"             // for Output, output
#include "rstFileOutput.hpp"      // for RstFileOutput
#include "trajectoryOutput.hpp"   // for TrajectoryOutput

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;
using namespace output;

/**
 * @brief Construct a new Input File Parser Output:: Input File Parser Output object
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
}

/**
 * @brief parse output frequency of simulation and set it in output statically
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

    Output::setOutputFrequency(size_t(outputFrequency));
}

/**
 * @brief parse log filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseLogFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getLogOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse info filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseInfoFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getInfoOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse energy filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseEnergyFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getEnergyOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse trajectory filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseTrajectoryFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getXyzOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseVelocityFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getVelOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse velocity filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseForceFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getForceOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse restart filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseRestartFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getRstFileOutput().setFilename(lineElements[2]);
}

/**
 * @brief parse charge filename of simulation and add it to output
 *
 * @param lineElements
 */
void InputFileParserOutput::parseChargeFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getChargeOutput().setFilename(lineElements[2]);
}