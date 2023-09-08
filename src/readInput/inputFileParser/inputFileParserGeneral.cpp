#include "inputFileParserGeneral.hpp"

#include "exceptions.hpp"   // for InputFileException, customException
#include "settings.hpp"     // for Settings

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new Input File Parser General:: Input File Parser General object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) jobtype <string> (required)
 *
 * @param engine
 */
InputFileParserGeneral::InputFileParserGeneral(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("jobtype"), bind_front(&InputFileParserGeneral::parseJobType, this), true);
}

/**
 * @brief parse jobtype of simulation and set it in settings
 *
 * @details Possible options are:
 * 1) mm-md
 *
 * @param lineElements
 *
 * @throw customException::InputFileException if jobtype is not recognised
 */
void InputFileParserGeneral::parseJobType(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "mm-md")
        settings::Settings::setJobtype("MMMD");
    else
        throw customException::InputFileException(
            format("Invalid jobtype \"{}\" at line {} in input file", lineElements[2], lineNumber));
}