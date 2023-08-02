#include "constants.hpp"
#include "inputFileReader.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace manostat;
using namespace customException;

/**
 * @brief Parse the manostat used in the simulation
 *
 * @param lineElements
 */
void InputFileReader::parseManostat(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    if (lineElements[2] == "none")
    {
        _engine.makeManostat(Manostat());
        _engine.getSettings().setManostat("none");
    }
    else if (lineElements[2] == "berendsen")
    {
        _engine.makeManostat(BerendsenManostat());
        _engine.getSettings().setManostat("berendsen");
    }
    else
        throw InputFileException("Invalid manostat \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file");
}

/**
 * @brief Parse the pressure used in the simulation
 *
 * @param lineElements
 */
void InputFileReader::parsePressure(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setPressure(stod(lineElements[2]));
}

/**
 * @brief parses the relaxation time of the manostat
 *
 * @param lineElements
 */
void InputFileReader::parseManostatRelaxationTime(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    _engine.getSettings().setTauManostat(stod(lineElements[2]));
}

/**
 * @brief Parse the compressibility used in the simulation (isothermal compressibility)
 *
 * @param lineElements
 *
 * @throw InputFileException if compressibility is negative
 */
void InputFileReader::parseCompressibility(const vector<string> &lineElements)
{
    checkCommand(lineElements, _lineNumber);
    const auto compressibility = stod(lineElements[2]);
    if (compressibility < 0.0)
        throw InputFileException("Invalid compressibility \"" + lineElements[2] + "\" at line " + to_string(_lineNumber) +
                                 "in input file - has to be positive");

    _engine.getSettings().setCompressibility(compressibility);
}