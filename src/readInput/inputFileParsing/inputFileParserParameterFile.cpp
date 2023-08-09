#include "exceptions.hpp"
#include "inputFileParser.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Parameter File:: Input File Parser Parameter File object
 *
 * @param engine
 */
InputFileParserParameterFile::InputFileParserParameterFile(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("parameter_file"), bind_front(&InputFileParserParameterFile::parseParameterFilename, this), false);
}

/**
 * @brief parse parameter file name of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserParameterFile::parseParameterFilename(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (filename.empty()) throw InputFileException("Parameter filename cannot be empty");

    if (!utilities::fileExists(filename)) throw InputFileException("Cannot open parameter file - filename = " + string(filename));

    _engine.getSettings().setParameterFilename(filename);
}