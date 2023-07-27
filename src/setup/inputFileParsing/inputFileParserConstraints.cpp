#include "exceptions.hpp"
#include "inputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace customException;

/**
 * @brief parsing if shake is activated
 *
 * @details only "on" and "off" (default) are valid keywords
 *
 * @param lineElements
 */
void InputFileReader::parseShakeActivated(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "on")
        _engine.getConstraints().activate();
    else if (lineElements[2] == "off")
    {
        // default
    }
    else
    {
        auto message  = "Invalid shake keyword \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) + "in input file\n";
        message      += R"(Possible keywords are "on" and "off")";
        throw InputFileException(message);
    }
}

/**
 * @brief parsing shake tolerance
 *
 * @param lineElements
 */
void InputFileReader::parseShakeTolerance(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);

    auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
    {
        auto message = "Invalid shake tolerance \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
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
void InputFileReader::parseShakeIteration(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);

    auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
    {
        auto message = "Invalid shake iteration \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
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
void InputFileReader::parseRattleTolerance(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);

    auto tolerance = stod(lineElements[2]);

    if (tolerance < 0.0)
    {
        auto message = "Invalid rattle tolerance \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
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
void InputFileReader::parseRattleIteration(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);

    auto iteration = stoi(lineElements[2]);

    if (iteration < 0)
    {
        auto message = "Invalid rattle iteration \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                       "in input file - Iteration must be positive\n";
        throw InputFileException(message);
    }

    _engine.getSettings().setRattleMaxIter(iteration);
}