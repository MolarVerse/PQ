#include "exceptions.hpp"
#include "inputFileParser.hpp"

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
 */
void InputFileParserConstraints::parseShakeActivated(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "on")
        _engine.getConstraints().activate();
    else if (lineElements[2] == "off")
    {
        // default
    }
    else
    {
        auto message  = "Invalid shake keyword \"" + lineElements[2] + "\" at line " + to_string(lineNumber) + "in input file\n";
        message      += R"(Possible keywords are "on" and "off")";
        throw InputFileException(message);
    }
}

/**
 * @brief parsing shake tolerance
 *
 * @param lineElements
 */
void InputFileParserConstraints::parseShakeTolerance(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
    {
        auto message = "Invalid shake tolerance \"" + lineElements[2] + "\" at line " + to_string(lineNumber) +
                       "in input file - Tolerance must be positive\n";
        throw InputFileException(message);
    }

    _engine.getSettings().setShakeTolerance(stod(lineElements[2]));
}

/**
 * @brief parsing shake iteration
 *
 * @param lineElements
 */
void InputFileParserConstraints::parseShakeIteration(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
    {
        auto message = "Invalid shake iteration \"" + lineElements[2] + "\" at line " + to_string(lineNumber) +
                       "in input file - Iteration must be positive\n";
        throw InputFileException(message);
    }

    _engine.getSettings().setShakeMaxIter(iteration);
}

/**
 * @brief parsing rattle tolerance
 *
 * @param lineElements
 */
void InputFileParserConstraints::parseRattleTolerance(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
    {
        auto message = "Invalid rattle tolerance \"" + lineElements[2] + "\" at line " + to_string(lineNumber) +
                       "in input file - Tolerance must be positive\n";
        throw InputFileException(message);
    }

    _engine.getSettings().setRattleTolerance(tolerance);
}

/**
 * @brief parsing rattle iteration
 *
 * @param lineElements
 */
void InputFileParserConstraints::parseRattleIteration(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
    {
        auto message = "Invalid rattle iteration \"" + lineElements[2] + "\" at line " + to_string(lineNumber) +
                       "in input file - Iteration must be positive\n";
        throw InputFileException(message);
    }

    _engine.getSettings().setRattleMaxIter(iteration);
}