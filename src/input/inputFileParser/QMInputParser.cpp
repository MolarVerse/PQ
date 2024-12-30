/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "QMInputParser.hpp"

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "exceptions.hpp"         // for InputFileException, customException
#include "qmSettings.hpp"         // for Settings
#include "references.hpp"         // for ReferencesOutput
#include "referencesOutput.hpp"   // for ReferencesOutput
#include "stringUtilities.hpp"    // for toLowerCopy

using namespace input;
using namespace utilities;
using namespace settings;
using namespace customException;
using namespace engine;
using namespace references;

/**
 * @brief Construct a new QMInputParser:: QMInputParser object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) qm_prog <string> 2) qm_script
 * <string>
 *
 * @param engine
 */
QMInputParser::QMInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("qm_prog"),
        bind_front(&QMInputParser::parseQMMethod, this),
        false
    );

    addKeyword(
        std::string("qm_script"),
        bind_front(&QMInputParser::parseQMScript, this),
        false
    );

    addKeyword(
        std::string("qm_script_full_path"),
        bind_front(&QMInputParser::parseQMScriptFullPath, this),
        false
    );

    addKeyword(
        std::string("qm_loop_time_limit"),
        bind_front(&QMInputParser::parseQMLoopTimeLimit, this),
        false
    );

    addKeyword(
        std::string("dispersion"),
        bind_front(&QMInputParser::parseDispersion, this),
        false
    );

    addKeyword(
        std::string("mace_model_size"),
        bind_front(&QMInputParser::parseMaceModelSize, this),
        false
    );
}

/**
 * @brief parse external QM Program which should be used
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the method is not recognized
 */
void QMInputParser::parseQMMethod(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum QMMethod;
    checkCommand(lineElements, lineNumber);

    const auto method = toLowerAndReplaceDashesCopy(lineElements[2]);

    if ("dftbplus" == method)
    {
        QMSettings::setQMMethod(DFTBPLUS);
        ReferencesOutput::addReferenceFile(_DFTBPLUS_FILE_);
    }
    else if ("pyscf" == method)
    {
        QMSettings::setQMMethod(PYSCF);
        ReferencesOutput::addReferenceFile(_PYSCF_FILE_);
    }

    else if ("turbomole" == method)
    {
        QMSettings::setQMMethod(TURBOMOLE);
        ReferencesOutput::addReferenceFile(_TURBOMOLE_FILE_);
    }

    else if (method.starts_with("mace"))
        parseMaceQMMethod(method);

    else
        throw InputFileException(std::format(
            "Invalid qm_prog \"{}\" in input file.\n"
            "Possible values are: dftbplus, pyscf, turbomole, mace, mace_mp, "
            "mace_off",
            lineElements[2]
        ));
}

/**
 * @brief parse external QM Script name
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseQMScript(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMSettings::setQMScript(lineElements[2]);
}

/**
 * @brief parse external QM script name
 *
 * @details this keyword is used for singularity builds to ensure that the user
 * knows what he is doing. With a singularity build the script has to be
 * accessed from outside of the container and therefore the general keyword
 * qm_script is not applicable.
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseQMScriptFullPath(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMSettings::setQMScriptFullPath(lineElements[2]);
}

/**
 * @brief parse the time limit for the QM loop
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseQMLoopTimeLimit(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMSettings::setQMLoopTimeLimit(std::stod(lineElements[2]));
}

/**
 * @brief parse the dispersion correction
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseDispersion(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto dispersion = toLowerCopy(lineElements[2]);

    if ("true" == dispersion || "on" == dispersion)
        QMSettings::setUseDispersionCorrection(true);

    else if ("false" == dispersion || "off" == dispersion)
        QMSettings::setUseDispersionCorrection(false);

    else
        throw InputFileException(std::format(
            "Invalid dispersion \"{}\" in input file.\n"
            "Possible values are: true, false, on, off",
            lineElements[2]
        ));
}

/**
 * @brief parse the size of the Mace model
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the size is not recognized
 */
void QMInputParser::parseMaceModelSize(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum MaceModelSize;
    checkCommand(lineElements, lineNumber);

    const auto size = toLowerCopy(lineElements[2]);

    if ("small" == size)
        QMSettings::setMaceModelSize(SMALL);

    else if ("medium" == size)
        QMSettings::setMaceModelSize(MEDIUM);

    else if ("large" == size)
        QMSettings::setMaceModelSize(LARGE);

    else
        throw InputFileException(std::format(
            "Invalid mace_model_size \"{}\" in input file.\n"
            "Possible values are: small, medium, large",
            lineElements[2]
        ));
}

/**
 * @brief parses the QM method if it starts with "mace"
 *
 * @param model
 *
 * @throws InputFileException if the model is not recognized
 */
void QMInputParser::parseMaceQMMethod(const std::string_view &model)
{
    using enum MaceModelType;

    if ("mace" == model || "mace_mp" == model)
    {
        QMSettings::setMaceModelType(MACE_MP);
        ReferencesOutput::addReferenceFile(_MACEMP_FILE_);
    }

    else if ("mace_off" == model)
    {
        QMSettings::setMaceModelType(MACE_OFF);
        ReferencesOutput::addReferenceFile(_MACEOFF_FILE_);
    }

    else if ("mace_anicc" == model || "mace_ani" == model)
        throw InputFileException(std::format(
            "The mace ani model is not supported in this version of PQ.\n"
        ));

    else
    {
        throw InputFileException(std::format(
            "Invalid mace type qm_method \"{}\" in input file.\n"
            "Possible values are: mace (mace_mp), mace_off",
            model
        ));
    }

    QMSettings::setQMMethod(QMMethod::MACE);
}