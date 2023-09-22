#include "inputFileParserCoulombLongRange.hpp"

#include "exceptions.hpp"          // for InputFileException, customException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "stringUtilities.hpp"     // for toLowerCopy

#include <cstddef>       // for size_t, std
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Coulomb Long Range:: Input File Parser Coulomb Long Range object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) long_range <string>
 * 2) wolf_param <double>
 *
 * @param engine
 */
InputFileParserCoulombLongRange::InputFileParserCoulombLongRange(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("long_range"), bind_front(&InputFileParserCoulombLongRange::parseCoulombLongRange, this), false);
    addKeyword(std::string("wolf_param"), bind_front(&InputFileParserCoulombLongRange::parseWolfParameter, this), false);
}

/**
 * @brief Parse the coulombic long-range correction used in the simulation
 *
 * @details Possible options are:
 * 1) "none" - no long-range correction is used (default) = shifted potential
 * 2) "wolf" - wolf long-range correction is used
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if coulombic long-range correction is not valid - currently only none and wolf are
 * supported
 */
void InputFileParserCoulombLongRange::parseCoulombLongRange(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto type = utilities::toLowerCopy(lineElements[2]);

    if (type == "none")
        settings::PotentialSettings::setCoulombLongRangeType("none");

    else if (type == "wolf")
        settings::PotentialSettings::setCoulombLongRangeType("wolf");

    else
        throw customException::InputFileException(format(
            R"(Invalid long-range type for coulomb correction "{}" at line {} in input file - possible options are "none", "wolf")",
            type,
            lineNumber));
}

/**
 * @brief parse the wolf parameter used in the simulation
 *
 * @details default value is 0.25
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if wolf parameter is negative
 */
void InputFileParserCoulombLongRange::parseWolfParameter(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto wolfParameter = stod(lineElements[2]);

    if (wolfParameter < 0.0)
        throw customException::InputFileException("Wolf parameter cannot be negative");

    settings::PotentialSettings::setWolfParameter(wolfParameter);
}