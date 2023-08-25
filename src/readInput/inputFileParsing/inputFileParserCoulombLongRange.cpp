#include "inputFileParserCoulombLongRange.hpp"

#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException, customException
#include "settings.hpp"     // for Settings
#include "thermostat.hpp"   // for thermostat

#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace std;
using namespace readInput;
using namespace thermostat;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Coulomb Long Range:: Input File Parser Coulomb Long Range object
 *
 * @param engine
 */
InputFileParserCoulombLongRange::InputFileParserCoulombLongRange(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("long_range"), bind_front(&InputFileParserCoulombLongRange::parseCoulombLongRange, this), false);
    addKeyword(string("wolf_param"), bind_front(&InputFileParserCoulombLongRange::parseWolfParameter, this), false);
}

/**
 * @brief Parse the coulombic long-range correction used in the simulation
 *
 * @details possible options are "none" and "wolf"
 *
 * @param lineElements
 *
 * @throws InputFileException if coulombic long-range correction is not valid - currently only none and wolf are supported
 */
void InputFileParserCoulombLongRange::parseCoulombLongRange(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "none")
        _engine.getSettings().setCoulombLongRangeType("none");
    else if (lineElements[2] == "wolf")
        _engine.getSettings().setCoulombLongRangeType("wolf");
    else
        throw InputFileException(format(
            R"(Invalid long-range type for coulomb correction "{}" at line {} in input file)", lineElements[2], lineNumber));
}

/**
 * @brief parse the wolf parameter used in the simulation
 *
 * @param lineElements
 *
 * @throws InputFileException if wolf parameter is negative
 */
void InputFileParserCoulombLongRange::parseWolfParameter(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    auto wolfParameter = stod(lineElements[2]);
    if (wolfParameter < 0.0)
        throw InputFileException("Wolf parameter cannot be negative");

    _engine.getSettings().setWolfParameter(wolfParameter);
}
