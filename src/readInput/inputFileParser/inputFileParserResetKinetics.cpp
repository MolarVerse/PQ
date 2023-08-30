#include "inputFileParserResetKinetics.hpp"

#include "exceptions.hpp"              // for InputFileException, customException
#include "resetKineticsSettings.hpp"   // for ResetKineticsSettings

#include <cstddef>       // for size_t, std
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Reset Kinetics:: Input File Parser Reset Kinetics object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) nscale <size_t>
 * 2) fscale <size_t>
 * 3) nreset <size_t>
 * 4) freset <size_t>
 *
 * @param engine
 */
InputFileParserResetKinetics::InputFileParserResetKinetics(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("nscale"), bind_front(&InputFileParserResetKinetics::parseNScale, this), false);
    addKeyword(std::string("fscale"), bind_front(&InputFileParserResetKinetics::parseFScale, this), false);
    addKeyword(std::string("nreset"), bind_front(&InputFileParserResetKinetics::parseNReset, this), false);
    addKeyword(std::string("freset"), bind_front(&InputFileParserResetKinetics::parseFReset, this), false);
}

/**
 * @brief parse nscale and set it in settings
 *
 * @details default value is 0
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw customException::InputFileException if nscale is negative
 */
void InputFileParserResetKinetics::parseNScale(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto nScale = stoi(lineElements[2]);

    if (nScale < 0)
        throw customException::InputFileException("Nscale must be positive");

    settings::ResetKineticsSettings::setNScale(size_t(nScale));
}

/**
 * @brief parse fscale and set it in settings
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw customException::InputFileException if fscale is negative
 */
void InputFileParserResetKinetics::parseFScale(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto fScale = stoi(lineElements[2]);

    if (fScale < 0)
        throw customException::InputFileException("Fscale must be positive");

    settings::ResetKineticsSettings::setFScale(static_cast<size_t>(fScale));
}

/**
 * @brief parse nreset and set it in settings
 *
 * @details default value is 0
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw customException::InputFileException if nreset is negative
 */
void InputFileParserResetKinetics::parseNReset(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto nReset = stoi(lineElements[2]);

    if (nReset < 0)
        throw customException::InputFileException("Nreset must be positive");

    settings::ResetKineticsSettings::setNReset(static_cast<size_t>(nReset));
}

/**
 * @brief parse freset and set it in settings
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throw customException::InputFileException if freset is negative
 */
void InputFileParserResetKinetics::parseFReset(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto fReset = stoi(lineElements[2]);

    if (fReset < 0)
        throw customException::InputFileException("Freset must be positive");

    settings::ResetKineticsSettings::setFReset(static_cast<size_t>(fReset));
}
