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
    addKeyword(std::string("start_file"), bind_front(&InputFileParserGeneral::parseStartFilename, this), true);
    addKeyword(std::string("moldescriptor_file"), bind_front(&InputFileParserGeneral::parseMoldescriptorFilename, this), false);
    addKeyword(std::string("guff_path"), bind_front(&InputFileParserGeneral::parseGuffPath, this), false);
    addKeyword(std::string("guff_file"), bind_front(&InputFileParserGeneral::parseGuffDatFilename, this), false);
}

/**
 * @brief parse start file of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserGeneral::parseStartFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setStartFilename(lineElements[2]);
}

/**
 * @brief parse moldescriptor file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileParserGeneral::parseMoldescriptorFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setMoldescriptorFilename(lineElements[2]);
}

/**
 * @brief parse guff path of simulation and set it in settings
 *
 * @throws InputFileException deprecated keyword
 */
[[noreturn]] void InputFileParserGeneral::parseGuffPath(const std::vector<std::string> &, const size_t)
{
    throw InputFileException(R"(The "guff_path" keyword id deprecated. Please use "guffdat_file" instead.)");
}

/**
 * @brief parse guff dat file of simulation and set it in settings
 *
 * @param lineElements
 */
void InputFileParserGeneral::parseGuffDatFilename(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    _engine.getSettings().setGuffDatFilename(lineElements[2]);
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