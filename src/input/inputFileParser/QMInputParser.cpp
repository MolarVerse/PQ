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

#include <format>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "exceptions.hpp"
#include "hubbardDerivMap.hpp"
#include "parserUtils.hpp"
#include "qmSettings.hpp"
#include "references.hpp"
#include "referencesOutput.hpp"
#include "stdoutOutput.hpp"
#include "stringUtilities.hpp"

namespace input
{

    /**
     * @brief Construct a new QMInputParser:: QMInputParser object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) qm_prog "<string>" 2)
     * qm_script
     * "<string>"
     *
     * @param logOutput
     */
    QMInputParser::QMInputParser(out::LogOutput &logOutput)
        : QMInputParser(logOutput, true)
    {
    }

    /**
     * @brief Construct a new QMInputParser:: QMInputParser object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) qm_prog "<string>" 2)
     * qm_script
     * "<string>"
     *
     * @param logOutput
     * @param resolveBuiltInSlakosPath
     */
    QMInputParser::QMInputParser(
        out::LogOutput &logOutput,
        const bool      resolveBuiltInSlakosPath
    )
        : _logOutput(&logOutput),
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
     * @throws exc::InputFileException if the method is not recognized
     */
    void QMInputParser::parseQMMethod(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        using enum settings::QMMethod;
        checkCommand(lineElements, lineNumber);

        const auto method =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        if ("dftbplus" == method)
        {
            settings::QMSettings::setQMMethod(DFTBPLUS);
            references::ReferencesOutput::addReferenceFile(
                references::DFTBPLUS_FILE
            );
        }
        else if ("ase_dftbplus" == method)
        {
            settings::QMSettings::setQMMethod(ASEDFTBPLUS);
            references::ReferencesOutput::addReferenceFile(
                references::DFTBPLUS_FILE
            );
        }
        else if ("ase_xtb" == method)
        {
            settings::QMSettings::setQMMethod(ASEXTB);
        }
        else if ("pyscf" == method)
        {
            settings::QMSettings::setQMMethod(PYSCF);
            references::ReferencesOutput::addReferenceFile(
                references::PYSCF_FILE
            );
        }
        else if ("turbomole" == method)
        {
            settings::QMSettings::setQMMethod(TURBOMOLE);
            references::ReferencesOutput::addReferenceFile(
                references::TURBOMOLE_FILE
            );
        }
        else if ("fennol" == method)
        {
            settings::QMSettings::setQMMethod(method);
            references::ReferencesOutput::addReferenceFile(
                references::FENNOL_FILE
            );
        }
        else if (method.starts_with("mace"))
        {
            parseMaceQMMethod(method);
        }
        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid qm_prog \"{}\" in input file.\n"
                    "Possible values are: dftbplus, ase_dftbplus, ase_xtb, "
                    "pyscf, "
                    "turbomole, fennol, mace, mace_mp, mace_off",
                    lineElements[2]
                )
            );
        }
    }

    /**
     * @brief parse external QM Script name
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseQMScript(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setQMScript(lineElements[2]);
    }

    /**
     * @brief parse external QM script name
     *
     * @details this keyword is used for singularity builds to ensure that the
     * user knows what he is doing. With a singularity build the script has to
     * be accessed from outside of the container and therefore the general
     * keyword qm_script is not applicable.
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseQMScriptFullPath(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setQMScriptFullPath(lineElements[2]);
    }

    /**
     * @brief parse the time limit for the QM loop
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseQMLoopTimeLimit(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setQMLoopTimeLimit(
            utilities::stringToFiniteDouble(lineElements[2])
        );
    }

    /**
     * @brief parse the dispersion correction
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseDispersion(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setUseDispersionCorrection(
            utilities::keywordToBool(lineElements)
        );
    }

    /**
     * @brief parse the remove net force option
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseRemoveNetForce(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        settings::QMSettings::setRemoveNetForce(
            utilities::keywordToBool(lineElements)
        );
    }

    /**
     * @brief parse the Mace model
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if the model is not recognized
     */
    void QMInputParser::parseMaceModel(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        using enum settings::MaceModel;
        checkCommand(lineElements, lineNumber);

        const auto *const modelSizeWarning =
            "The keyword \"mace_model_size\" is deprecated and has been "
            "renamed to "
            "\"mace_model\". It will be removed in a future release.";

        if (lineElements[0] == "mace_model_size")
        {
            _logOutput->queueWarning(modelSizeWarning);
            out::StdoutOutput::writeSetupWarning(modelSizeWarning);
        }

        const auto size =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        if ("small" == size)
            settings::QMSettings::setMaceModel(SMALL);

        else if ("medium" == size)
            settings::QMSettings::setMaceModel(MEDIUM);

        else if ("large" == size)
            settings::QMSettings::setMaceModel(LARGE);

        else if ("small_0b" == size)
            settings::QMSettings::setMaceModel(SMALL0B);

        else if ("medium_0b" == size)
            settings::QMSettings::setMaceModel(MEDIUM0B);

        else if ("small_0b2" == size)
            settings::QMSettings::setMaceModel(SMALL0B2);

        else if ("medium_0b2" == size)
            settings::QMSettings::setMaceModel(MEDIUM0B2);

        else if ("large_0b2" == size)
            settings::QMSettings::setMaceModel(LARGE0B2);

        else if ("medium_0b3" == size)
            settings::QMSettings::setMaceModel(MEDIUM0B3);

        else if ("medium_mpa_0" == size)
            settings::QMSettings::setMaceModel(MEDIUMMPA0);

        else if ("medium_omat_0" == size)
            settings::QMSettings::setMaceModel(MEDIUMOMAT0);

        else if ("custom" == size)
            settings::QMSettings::setMaceModel(CUSTOM);

        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid mace_model \"{}\" in input file.\n"
                    "Possible values are: small, medium, large, small-0b,\n"
                    "medium-0b, small-0b2, medium-0b2, large-0b2, medium-0b3,\n"
                    "medium-mpa-0, medium-omat-0, custom",
                    lineElements[2]
                )
            );
        }
    }

    /**
     * @brief parse the MACE evaluation mode
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseMaceMode(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        settings::QMSettings::setMaceMode(lineElements[2]);
    }

    /**
     * @brief parse external MACE model url
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseMaceModelPath(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setMaceModelPath(lineElements[2]);
    }

    /**
     * @brief parses the QM method if it starts with "mace"
     *
     * @param model
     *
     * @throws exc::InputFileException if the model is not recognized
     */
    void QMInputParser::parseMaceQMMethod(const std::string_view &model)
    {
        using enum settings::MaceModelType;

        if ("mace" == model || "mace_mp" == model)
        {
            settings::QMSettings::setMaceModelType(MACE_MP);
            references::ReferencesOutput::addReferenceFile(
                references::MACEMP_FILE
            );
        }

        else if ("mace_off" == model)
        {
            settings::QMSettings::setMaceModelType(MACE_OFF);
            references::ReferencesOutput::addReferenceFile(
                references::MACEOFF_FILE
            );
        }

        else if ("mace_anicc" == model || "mace_ani" == model)
        {
            throw exc::InputFileException(
                std::format(
                    "The mace ani model is not supported in this version of "
                    "PQ.\n"
                )
            );
        }
        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid mace type qm_method \"{}\" in input file.\n"
                    "Possible values are: mace (mace_mp), mace_off",
                    model
                )
            );
        }

        settings::QMSettings::setQMMethod(settings::QMMethod::MACE);
    }

    /**
     * @brief parse the Slakos type to be used
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if the slakos type is not recognized
     */
    void QMInputParser::parseSlakosType(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    ) const
    {
        using enum settings::SlakosType;
        checkCommand(lineElements, lineNumber);

        const auto slakos = utilities::toLowerCopy(lineElements[2]);

        if ("3ob" == slakos)
        {
            settings::QMSettings::setSlakosType(
                THREEOB,
                _resolveBuiltInSlakosPath
            );
            settings::QMSettings::setHubbardDerivs(hubbardDerivMap3ob);
            references::ReferencesOutput::addReferenceFile(
                references::THREEOB_FILE
            );
        }

        else if ("matsci" == slakos)
        {
            settings::QMSettings::setSlakosType(
                MATSCI,
                _resolveBuiltInSlakosPath
            );
            references::ReferencesOutput::addReferenceFile(
                references::MATSCI_FILE
            );
        }

        else if ("custom" == slakos)
        {
            settings::QMSettings::setSlakosType(CUSTOM);
        }
        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid slakos type \"{}\" in input file.\n"
                    "Possible values are: 3ob, matsci, custom",
                    lineElements[2]
                )
            );
        }
    }

    /**
     * @brief parse external Slakos path
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseSlakosPath(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setSlakosPath(lineElements[2]);
    }

    /**
     * @brief parse if third order DFTB is used
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseThirdOrder(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        settings::QMSettings::setUseThirdOrderDftb(
            utilities::keywordToBool(lineElements)
        );
        settings::QMSettings::setIsThirdOrderDftbSet(true);
    }

    /**
     * @brief parse custom Hubbard Derivative dictionary
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseHubbardDerivs(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommandArray(lineElements, lineNumber);

        std::unordered_map<std::string, double> hubbardDerivs;
        std::string                             derivs;

        for (size_t i = 2; i < lineElements.size(); ++i)
        {
            derivs += lineElements[i];
        }

        std::stringstream sstream(derivs);
        std::string       item;
        while (std::getline(sstream, item, ','))
        {
            const auto separator = item.find(':');

            if (separator == std::string::npos || 0 == separator ||
                separator + 1 == item.size() ||
                item.find(':', separator + 1) != std::string::npos)
            {
                throw exc::InputFileException(
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
                    utilities::stringToFiniteDouble(item.substr(separator + 1));
            }
            catch (const std::invalid_argument &)
            {
                throw exc::InputFileException(
                    std::format(
                        "Invalid hubbard_derivs format \"{}\" in input file.",
                        derivs
                    )
                );
            }
            catch (const std::out_of_range &)
            {
                throw exc::InputFileException(
                    std::format(
                        "Invalid hubbard_derivs format \"{}\" in input file.",
                        derivs
                    )
                );
            }
        }

        settings::QMSettings::setHubbardDerivs(hubbardDerivs);
        settings::QMSettings::setIsHubbardDerivsSet(true);
    }

    /**
     * @brief parse the xTB method to be used
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if the xTB method is not recognized
     */
    void QMInputParser::parseXtbMethod(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        using enum settings::XtbMethod;
        checkCommand(lineElements, lineNumber);

        const auto slakos =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        if ("gfn1_xtb" == slakos)
            settings::QMSettings::setXtbMethod(GFN1);

        else if ("gfn2_xtb" == slakos)
            settings::QMSettings::setXtbMethod(GFN2);

        else if ("ipea1_xtb" == slakos)
            settings::QMSettings::setXtbMethod(IPEA1);

        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid xTB method \"{}\" in input file.\n"
                    "Possible values are: GFN1-xTB, GFN2-xTB, IPEA1-xTB",
                    lineElements[2]
                )
            );
        }
    }

    /**
     * @brief parse FeNNol model path
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseFennolModelPath(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setFennolModelPath(lineElements[2]);
    }

    /**
     * @brief parse if GPU pre-processing is enabled for FeNNol
     *
     * @param lineElements
     * @param lineNumber
     */
    void QMInputParser::parseGPUPreprocessing(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        settings::QMSettings::setUseGPUPreprocessing(
            utilities::keywordToBool(lineElements)
        );
    }

}   // namespace input
