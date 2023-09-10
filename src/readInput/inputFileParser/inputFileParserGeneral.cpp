#include "inputFileParserGeneral.hpp"

#include "exceptions.hpp"   // for InputFileException, customException
#include "mmmdEngine.hpp"   // for MMMDEngine
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
 * @brief parse jobtype of simulation left empty just to not parse it again after engine is generated
 */
void InputFileParserGeneral::parseJobType(const std::vector<std::string> &, const size_t) {}

/**
 * @brief parse jobtype of simulation and set it in settings and generate engine
 *
 * @details Possible options are:
 * 1) mm-md
 *
 * @param lineElements
 * @param lineNumber
 * @param engine
 *
 * @throw customException::InputFileException if jobtype is not recognised
 */
void InputFileParserGeneral::parseJobTypeForEngine(const std::vector<std::string>  &lineElements,
                                                   const size_t                     lineNumber,
                                                   std::unique_ptr<engine::Engine> &engine)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "mm-md")
    {
        settings::Settings::setJobtype("MMMD");
        engine.reset(new engine::MMMDEngine());
    }
    else
        throw customException::InputFileException(
            format("Invalid jobtype \"{}\" in input file - possible values are: mm-md", lineElements[2]));
}