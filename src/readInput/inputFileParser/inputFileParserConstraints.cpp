#include "inputFileParserConstraints.hpp"

#include "constraintSettings.hpp"   // for ConstraintSettings
#include "constraints.hpp"          // for Constraints
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for InputFileException, customException
#include "settings.hpp"             // for Settings

#include <cstddef>       // for size_t
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Constraints:: Input File Parser Constraints object
 *
 * @param engine
 */
InputFileParserConstraints::InputFileParserConstraints(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("shake"), bind_front(&InputFileParserConstraints::parseShakeActivated, this), false);
    addKeyword(string("shake-tolerance"), bind_front(&InputFileParserConstraints::parseShakeTolerance, this), false);
    addKeyword(string("shake-iter"), bind_front(&InputFileParserConstraints::parseShakeIteration, this), false);
    addKeyword(string("rattle-iter"), bind_front(&InputFileParserConstraints::parseRattleIteration, this), false);
    addKeyword(string("rattle-tolerance"), bind_front(&InputFileParserConstraints::parseRattleTolerance, this), false);
}

/**
 * @brief parsing if shake is activated
 *
 * @details only "on" and "off" (default) are valid keywords
 *
 * @param lineElements
 *
 * @throws InputFileException if keyword is not valid - currently only on and off are supported
 */
void InputFileParserConstraints::parseShakeActivated(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "on")
        _engine.getConstraints().activate();
    else if (lineElements[2] == "off")
        _engine.getConstraints().deactivate();
    else
    {
        auto message = format(R"(Invalid shake keyword "{}" at line {} in input file\n Possible keywords are "on" and "off")",
                              lineElements[2],
                              lineNumber);
        throw InputFileException(message);
    }
}

/**
 * @brief parsing shake tolerance
 *
 * @param lineElements
 *
 * @throw InputFileException if tolerance is negative
 */
void InputFileParserConstraints::parseShakeTolerance(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
        throw InputFileException("Shake tolerance must be positive");

    settings::ConstraintSettings::setShakeTolerance(tolerance);
}

/**
 * @brief parsing shake iteration
 *
 * @param lineElements
 *
 * @throw InputFileException if iteration is negative
 */
void InputFileParserConstraints::parseShakeIteration(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
        throw InputFileException("Maximum shake iterations must be positive");

    settings::ConstraintSettings::setShakeMaxIter(size_t(iteration));
}

/**
 * @brief parsing rattle tolerance
 *
 * @param lineElements
 *
 * @throw InputFileException if tolerance is negative
 */
void InputFileParserConstraints::parseRattleTolerance(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
        throw InputFileException("Rattle tolerance must be positive");

    settings::ConstraintSettings::setRattleTolerance(tolerance);
}

/**
 * @brief parsing rattle iteration
 *
 * @param lineElements
 *
 * @throw InputFileException if iteration is negative
 */
void InputFileParserConstraints::parseRattleIteration(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
        throw InputFileException("Maximum rattle iterations must be positive");

    settings::ConstraintSettings::setRattleMaxIter(size_t(iteration));
}