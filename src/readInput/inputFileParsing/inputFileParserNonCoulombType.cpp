#include "inputFileParser.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser Non Coulomb Type object
 *
 * @param engine
 */
InputFileParserNonCoulombType::InputFileParserNonCoulombType(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("noncoulomb"), bind_front(&InputFileParserNonCoulombType::parseNonCoulombType, this), false);
}

/**
 * @brief Parse the nonCoulombic type of the guff.dat file
 *
 * @details possible options are "none", "lj" and "buck"
 *
 * @param lineElements
 *
 * @throws InputFileException if invalid nonCoulomb type
 */
void InputFileParserNonCoulombType::parseNonCoulombType(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "none")
        _engine.getSettings().setNonCoulombType("none");
    else if (lineElements[2] == "lj")
        _engine.getSettings().setNonCoulombType("lj");
    else if (lineElements[2] == "buck")
        _engine.getSettings().setNonCoulombType("buck");
    else
        throw InputFileException(
            format("Invalid nonCoulomb type \"{}\" at line {} in input file. Possible options are: lj, buck and none",
                   lineElements[2],
                   lineNumber));
}