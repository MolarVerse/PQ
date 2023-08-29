#include "inputFileParserGeneral.hpp"

#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException, customException
#include "settings.hpp"     // for Settings

#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

using namespace readInput;
using namespace customException;

/**
 * @brief Construct a new Input File Parser General:: Input File Parser General object
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
 * @param lineElements
 *
 * @throw InputFileException if jobtype is not recognised
 */
void InputFileParserGeneral::parseJobType(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "mm-md")
        _engine.getSettings().setJobtype("MMMD");
    else
        throw InputFileException(format("Invalid jobtype \"{}\" at line {} in input file", lineElements[2], lineNumber));
}