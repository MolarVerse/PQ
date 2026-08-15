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

#include <format>          // for format
#include <sstream>         // for stringstream
#include <stdexcept>       // for invalid_argument, out_of_range
#include <unordered_map>   // for unordered_map

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException, customException
#include "hubbardDerivMap.hpp"   // for hubbardDerivMap3ob
#include "parserUtils.hpp"
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
using namespace constants;

/**
 * @brief Construct a new QMInputParser:: QMInputParser object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) qm_prog <string> 2) qm_script
 * <string>
 *
 * @param engine
 * @param logOutput
 * @param stdoutOutput
 * @param resolveBuiltInSlakosPath
 */
QMInputParser::QMInputParser(
    Engine                               &engine,
    output::LogOutput                    &logOutput,
    output::StdoutOutput                 &stdoutOutput,
    const bool                            resolveBuiltInSlakosPath
)
    : InputFileParser(engine),
      _logOutput(&logOutput),
      _stdoutOutput(&stdoutOutput),
      _resolveBuiltInSlakosPath(resolveBuiltInSlakosPath)
{
    addKeyword(
        std::string("qm_prog"),
        bindMember(&QMInputParser::parseQMMethod, this),
        false
    );

    addKeyword(
        std::string("qm_script"),
        bindMember(&QMInputParser::parseQMScript, this),
        false
    );

    addKeyword(
        std::string("qm_script_full_path"),
        bindMember(&QMInputParser::parseQMScriptFullPath, this),
        false
    );

    addKeyword(
        std::string("qm_loop_time_limit"),
        bindMember(&QMInputParser::parseQMLoopTimeLimit, this),
        false
    );

    addKeyword(
        std::string("dispersion"),
        bindMember(&QMInputParser::parseDispersion, this),
        false
    );

    addKeyword(
        std::string("remove_net_force"),
        bindMember(&QMInputParser::parseRemoveNetForce, this),
        false
    );

    addKeyword(
        std::string("mace_model_size"),
        bindMember(&QMInputParser::parseMaceModel, this),
        false
    );

    addKeyword(
        std::string("mace_model"),
        bindMember(&QMInputParser::parseMaceModel, this),
        false
    );

    addKeyword(
        std::string("mace_mode"),
        bindMember(&QMInputParser::parseMaceMode, this),
        false
    );

    addKeyword(
        std::string("mace_model_path"),
        bindMember(&QMInputParser::parseMaceModelPath, this),
        false
    );

    addKeyword(
        std::string("slakos"),
        bindMember(&QMInputParser::parseSlakosType, this),
        false
    );

    addKeyword(
        std::string("slakos_path"),
        bindMember(&QMInputParser::parseSlakosPath, this),
        false
    );

    addKeyword(
        std::string("third_order"),
        bindMember(&QMInputParser::parseThirdOrder, this),
        false
    );

    addKeyword(
        std::string("hubbard_derivs"),
        bindMember(&QMInputParser::parseHubbardDerivs, this),
        false
    );

    addKeyword(
        std::string("xtb_method"),
        bindMember(&QMInputParser::parseXtbMethod, this),
        false
    );

    addKeyword(
        std::string("fennol_model_path"),
        bindMember(&QMInputParser::parseFennolModelPath, this),
        false
    );

    addKeyword(
        std::string("gpu_preprocessing"),
        bindMember(&QMInputParser::parseGPUPreprocessing, this),
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
        ReferencesOutput::addReferenceFile(DFTBPLUS_FILE);
    }

    else if ("ase_dftbplus" == method)
    {
        QMSettings::setQMMethod(ASEDFTBPLUS);
        ReferencesOutput::addReferenceFile(DFTBPLUS_FILE);
    }

    else if ("ase_xtb" == method)
        QMSettings::setQMMethod(ASEXTB);

    else if ("pyscf" == method)
    {
        QMSettings::setQMMethod(PYSCF);
        ReferencesOutput::addReferenceFile(PYSCF_FILE);
    }

    else if ("turbomole" == method)
    {
        QMSettings::setQMMethod(TURBOMOLE);
        ReferencesOutput::addReferenceFile(TURBOMOLE_FILE);
    }

    else if ("fennol" == method)
    {
        QMSettings::setQMMethod(method);
        ReferencesOutput::addReferenceFile(FENNOL_FILE);
    }

    else if (method.starts_with("mace"))
        parseMaceQMMethod(method);

    else
        throw InputFileException(
            std::format(
                "Invalid qm_prog \"{}\" in input file.\n"
                "Possible values are: dftbplus, ase_dftbplus, ase_xtb, pyscf, "
                "turbomole, fennol, mace, mace_mp, mace_off",
                lineElements[2]
            )
        );
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
    QMSettings::setQMLoopTimeLimit(stringToFiniteDouble(lineElements[2]));
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
    QMSettings::setUseDispersionCorrection(keywordToBool(lineElements));
}

/**
 * @brief parse the remove net force option
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseRemoveNetForce(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    QMSettings::setRemoveNetForce(keywordToBool(lineElements));
}

/**
 * @brief parse the Mace model
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the model is not recognized
 */
void QMInputParser::parseMaceModel(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum MaceModel;
    checkCommand(lineElements, lineNumber);

    const auto modelSizeWarning =
        "The keyword \"mace_model_size\" is deprecated and has been renamed to "
        "\"mace_model\". It will be removed in a future release.";

    if (lineElements[0] == "mace_model_size")
    {
        _logOutput->queueWarning(modelSizeWarning);
        _stdoutOutput->writeSetupWarning(modelSizeWarning);
    }

    const auto size = toLowerAndReplaceDashesCopy(lineElements[2]);

    if ("small" == size)
        QMSettings::setMaceModel(SMALL);

    else if ("medium" == size)
        QMSettings::setMaceModel(MEDIUM);

    else if ("large" == size)
        QMSettings::setMaceModel(LARGE);

    else if ("small_0b" == size)
        QMSettings::setMaceModel(SMALL0B);

    else if ("medium_0b" == size)
        QMSettings::setMaceModel(MEDIUM0B);

    else if ("small_0b2" == size)
        QMSettings::setMaceModel(SMALL0B2);

    else if ("medium_0b2" == size)
        QMSettings::setMaceModel(MEDIUM0B2);

    else if ("large_0b2" == size)
        QMSettings::setMaceModel(LARGE0B2);

    else if ("medium_0b3" == size)
        QMSettings::setMaceModel(MEDIUM0B3);

    else if ("medium_mpa_0" == size)
        QMSettings::setMaceModel(MEDIUMMPA0);

    else if ("medium_omat_0" == size)
        QMSettings::setMaceModel(MEDIUMOMAT0);

    else if ("custom" == size)
        QMSettings::setMaceModel(CUSTOM);

    else
        throw InputFileException(
            std::format(
                "Invalid mace_model \"{}\" in input file.\n"
                "Possible values are: small, medium, large, small-0b,\n"
                "medium-0b, small-0b2, medium-0b2, large-0b2, medium-0b3,\n"
                "medium-mpa-0, medium-omat-0, custom",
                lineElements[2]
            )
        );
}

/**
 * @brief parse the MACE evaluation mode
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseMaceMode(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    QMSettings::setMaceMode(lineElements[2]);
}

/**
 * @brief parse external MACE model url
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseMaceModelPath(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMSettings::setMaceModelPath(lineElements[2]);
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
        ReferencesOutput::addReferenceFile(MACEMP_FILE);
    }

    else if ("mace_off" == model)
    {
        QMSettings::setMaceModelType(MACE_OFF);
        ReferencesOutput::addReferenceFile(MACEOFF_FILE);
    }

    else if ("mace_anicc" == model || "mace_ani" == model)
        throw InputFileException(
            std::format(
                "The mace ani model is not supported in this version of PQ.\n"
            )
        );

    else
    {
        throw InputFileException(
            std::format(
                "Invalid mace type qm_method \"{}\" in input file.\n"
                "Possible values are: mace (mace_mp), mace_off",
                model
            )
        );
    }

    QMSettings::setQMMethod(QMMethod::MACE);
}

/**
 * @brief parse the Slakos type to be used
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the slakos type is not recognized
 */
void QMInputParser::parseSlakosType(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum SlakosType;
    checkCommand(lineElements, lineNumber);

    const auto slakos = toLowerCopy(lineElements[2]);

    if ("3ob" == slakos)
    {
        QMSettings::setSlakosType(THREEOB, _resolveBuiltInSlakosPath);
        QMSettings::setHubbardDerivs(hubbardDerivMap3ob);
        ReferencesOutput::addReferenceFile(THREEOB_FILE);
    }

    else if ("matsci" == slakos)
    {
        QMSettings::setSlakosType(MATSCI, _resolveBuiltInSlakosPath);
        ReferencesOutput::addReferenceFile(MATSCI_FILE);
    }

    else if ("custom" == slakos)
        QMSettings::setSlakosType(CUSTOM);

    else
        throw InputFileException(
            std::format(
                "Invalid slakos type \"{}\" in input file.\n"
                "Possible values are: 3ob, matsci, custom",
                lineElements[2]
            )
        );
}

/**
 * @brief parse external Slakos path
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseSlakosPath(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMSettings::setSlakosPath(lineElements[2]);
}

/**
 * @brief parse if third order DFTB is used
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseThirdOrder(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    QMSettings::setUseThirdOrderDftb(keywordToBool(lineElements));
    QMSettings::setIsThirdOrderDftbSet(true);
}

/**
 * @brief parse custom Hubbard Derivative dictionary
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseHubbardDerivs(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommandArray(lineElements, lineNumber);

    std::unordered_map<std::string, double> hubbardDerivs;
    std::string                             derivs;

    for (size_t i = 2; i < lineElements.size(); ++i)
    {
        derivs += lineElements[i];
    }

    std::stringstream ss(derivs);
    std::string       item;
    while (std::getline(ss, item, ','))
    {
        const auto separator = item.find(':');

        if (separator == std::string::npos || 0 == separator ||
            separator + 1 == item.size() ||
            item.find(':', separator + 1) != std::string::npos)
        {
            throw InputFileException(
                std::format(
                    "Invalid hubbard_derivs format \"{}\" in input file.",
                    derivs
                )
            );
        }

        const auto element = item.substr(0, separator);
        try
        {
            hubbardDerivs[element] =
                stringToFiniteDouble(item.substr(separator + 1));
        }
        catch (const std::invalid_argument &)
        {
            throw InputFileException(
                std::format(
                    "Invalid hubbard_derivs format \"{}\" in input file.",
                    derivs
                )
            );
        }
        catch (const std::out_of_range &)
        {
            throw InputFileException(
                std::format(
                    "Invalid hubbard_derivs format \"{}\" in input file.",
                    derivs
                )
            );
        }
    }

    QMSettings::setHubbardDerivs(hubbardDerivs);
    QMSettings::setIsHubbardDerivsSet(true);
}

/**
 * @brief parse the xTB method to be used
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the xTB method is not recognized
 */
void QMInputParser::parseXtbMethod(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum XtbMethod;
    checkCommand(lineElements, lineNumber);

    const auto slakos = toLowerAndReplaceDashesCopy(lineElements[2]);

    if ("gfn1_xtb" == slakos)
        QMSettings::setXtbMethod(GFN1);

    else if ("gfn2_xtb" == slakos)
        QMSettings::setXtbMethod(GFN2);

    else if ("ipea1_xtb" == slakos)
        QMSettings::setXtbMethod(IPEA1);

    else
        throw InputFileException(
            std::format(
                "Invalid xTB method \"{}\" in input file.\n"
                "Possible values are: GFN1-xTB, GFN2-xTB, IPEA1-xTB",
                lineElements[2]
            )
        );
}

/**
 * @brief parse FeNNol model path
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseFennolModelPath(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMSettings::setFennolModelPath(lineElements[2]);
}

/**
 * @brief parse if GPU pre-processing is enabled for FeNNol
 *
 * @param lineElements
 * @param lineNumber
 */
void QMInputParser::parseGPUPreprocessing(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    QMSettings::setUseGPUPreprocessing(keywordToBool(lineElements));
}
