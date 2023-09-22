#include "inputFileParserGeneral.hpp"

#include "engine.hpp"                  // for Engine
#include "exceptions.hpp"              // for InputFileException, customException
#include "mmmdEngine.hpp"              // for MMMDEngine
#include "qmmdEngine.hpp"              // for QMMDEngine
#include "ringPolymerqmmdEngine.hpp"   // for RingPolymerQMMDEngine
#include "settings.hpp"                // for Settings
#include "stringUtilities.hpp"         // for toLowerCopy

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
 * @brief parse jobtype of simulation and set it in settings and reset engine unique_ptr
 *
 * @details Possible options are:
 * 1) mm-md
 * 2) qm-md
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

    const auto jobtype = utilities::toLowerCopy(lineElements[2]);

    if (jobtype == "mm-md")
    {
        settings::Settings::setJobtype("MMMD");
        settings::Settings::activateMM();
        engine.reset(new engine::MMMDEngine());
    }
    else if (jobtype == "qm-md")
    {
        settings::Settings::setJobtype("QMMD");
        settings::Settings::activateQM();
        engine.reset(new engine::QMMDEngine());
    }
    else if (jobtype == "qm-rpmd")
    {
        settings::Settings::setJobtype("RingPolymerQMMD");
        settings::Settings::activateQM();
        settings::Settings::activateRingPolymerMD();
        engine.reset(new engine::RingPolymerQMMDEngine());
    }
    else
        throw customException::InputFileException(
            format("Invalid jobtype \"{}\" in input file - possible values are: mm-md, qm-md, qm-rpmd", jobtype));
}