#include "inputFileParserTimings.hpp"

#include "exceptions.hpp"        // for InputFileException
#include "timingsSettings.hpp"   // for TimingsSettings

#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Timings:: Input File Parser Timings object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) timestep <double> (required)
 * 2) nstep <size_t> (required)
 *
 * @param engine
 */
InputFileParserTimings::InputFileParserTimings(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("timestep"), bind_front(&InputFileParserTimings::parseTimeStep, this), true);
    addKeyword(std::string("nstep"), bind_front(&InputFileParserTimings::parseNumberOfSteps, this), true);
}

/**
 * @brief parse timestep of simulation and set it in timings
 *
 * @param lineElements
 */
void InputFileParserTimings::parseTimeStep(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    settings::TimingsSettings::setTimeStep(stod(lineElements[2]));
}

/**
 * @brief parse number of steps of simulation and set it in timings
 *
 * @param lineElements
 *
 * @throws InputFileException if number of steps is negative
 */
void InputFileParserTimings::parseNumberOfSteps(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto numberOfSteps = stoi(lineElements[2]);

    if (numberOfSteps < 0)
        throw customException::InputFileException("Number of steps cannot be negative");

    settings::TimingsSettings::setNumberOfSteps(size_t(numberOfSteps));
}