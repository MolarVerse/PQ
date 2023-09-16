#include "inputFileParserQM.hpp"

#include "exceptions.hpp"   // for InputFileException, customException
#include "qmSettings.hpp"   // for Settings

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new InputFileParserQM:: InputFileParserQM object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) qm_prog <string>
 * 2) qm_script <string>
 *
 * @param engine
 */
InputFileParserQM::InputFileParserQM(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("qm_prog"), bind_front(&InputFileParserQM::parseQMMethod, this), false);
    addKeyword(std::string("qm_script"), bind_front(&InputFileParserQM::parseQMScript, this), false);
}

/**
 * @brief parse external QM Program which should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQM::parseQMMethod(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    if ("dftbplus" == lineElements[2])
        settings::QMSettings::setQMMethod("dftbplus");
    else
        throw customException::InputFileException(
            std::format("Invalid qm_prog \"{}\" in input file - possible values are: dftbplus", lineElements[2]));
}

/**
 * @brief parse external QM Script name
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserQM::parseQMScript(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    settings::QMSettings::setQMScript(lineElements[2]);
}