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
#include <mstd/string.hpp>
#include <unordered_map>

#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "hubbardDerivMap.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "qmSettings.hpp"
#include "references.hpp"
#include "referencesOutput.hpp"
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
     */
    QMInputParser::QMInputParser() : QMInputParser(true) {}

    /**
     * @brief Construct a new QMInputParser:: QMInputParser object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) qm_prog "<string>" 2)
     * qm_script
     * "<string>"
     *
     * @param resolveBuiltInSlakosPath
     */
    QMInputParser::QMInputParser(const bool resolveBuiltInSlakosPath)
        : _resolveBuiltInSlakosPath(resolveBuiltInSlakosPath)
    {
        addQMMethodKey();
        addQMScriptKey();
        addQMScriptFullPathKey();
        addQMLoopTimeLimitKey();
        addDispersionKey();
        addRemoveNetForceKey();
        addMaceModelKey();
        addMaceModeKey();
        addMaceModelPathKey();
        addSlakosTypeKey();
        addSlakosPathKey();
        addThirdOrderKey();
        addHubbardDerivsKey();
        addXtbMethodKey();
        addFennolModelPathKey();
        addGPUPreprocessingKey();
    }

    void QMInputParser::addQMMethodKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "qm_prog",
            .title = "QM Method",
            .description =
                "Specifies the quantum mechanical method to be used.",
        };

        const auto setValue = [](QMMethod value)
        {
            settings::QMSettings::setQMMethod(value);

            switch (value)
            {
                case QMMethod::ASE_DFTBPLUS:
                case QMMethod::DFTBPLUS:
                    references::ReferencesOutput::addReferenceFile(
                        references::DFTBPLUS_FILE
                    );
                    break;
                case QMMethod::TURBOMOLE:
                    references::ReferencesOutput::addReferenceFile(
                        references::TURBOMOLE_FILE
                    );
                    break;
                case QMMethod::FENNOL:
                    references::ReferencesOutput::addReferenceFile(
                        references::FENNOL_FILE
                    );
                    break;
                case QMMethod::PYSCF:
                    references::ReferencesOutput::addReferenceFile(
                        references::PYSCF_FILE
                    );
                    break;
                case QMMethod::MACE:
                case QMMethod::ASE_XTB:
                case QMMethod::NONE: break;
            }
        };

        const auto finalize = [](const std::string &value, QMMethod method)
        {
            if (method == QMMethod::MACE)
                parseMaceQMMethod(value);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<QMMethod>{
                .metadata   = metaData,
                .notAllowed = {QMMethod::NONE},
                .onSet      = setValue,
                .finalizer  = finalize,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addQMScriptKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "qm_script",
            .title       = "QM Script",
            .description = "Specifies the external QM script to be used.",
        };

        const auto setValue = [](const std::string &value)
        { settings::QMSettings::setQMScript(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::string>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addQMScriptFullPathKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "qm_script_full_path",
            .title       = "QM Script Full Path",
            .description = "Specifies the full path to the external QM script.",
        };

        const auto setValue = [](const std::string &value)
        { settings::QMSettings::setQMScriptFullPath(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::string>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addQMLoopTimeLimitKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "qm_loop_time_limit",
            .title       = "QM Loop Time Limit",
            .description = "Specifies the time limit for the QM loop.",
        };

        const auto setValue = [](double value)
        { settings::QMSettings::setQMLoopTimeLimit(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addDispersionKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "dispersion",
            .title       = "Dispersion Correction",
            .description = "Specifies whether to use dispersion correction.",
        };

        const auto setValue = [](bool value)
        { settings::QMSettings::setUseDispersionCorrection(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addRemoveNetForceKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "remove_net_force",
            .title       = "Remove Net Force",
            .description = "Specifies whether to remove the net force.",
        };

        const auto setValue = [](bool value)
        { settings::QMSettings::setRemoveNetForce(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addMaceModelKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "mace_model",
            .title       = "MACE Model",
            .description = "Specifies the MACE model to use.",
        };

        const auto setValue = [](MaceModel value)
        { settings::QMSettings::setMaceModel(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<MaceModel>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addMaceModeKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "mace_mode",
            .title       = "MACE Mode",
            .description = "Specifies the MACE evaluation mode.",
        };

        const auto setValue = [](MaceMode value)
        { settings::QMSettings::setMaceMode(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<MaceMode>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addMaceModelPathKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "mace_model_path",
            .title       = "MACE Model Path",
            .description = "Specifies the path to the external MACE model.",
        };

        const auto setValue = [](const std::string &value)
        { settings::QMSettings::setMaceModelPath(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::string>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addSlakosTypeKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "slakos",
            .title       = "SLAKOS Type",
            .description = "Specifies the SLAKOS type to use.",
        };

        const auto setValue = [this](SlakosType value)
        {
            switch (value)
            {
                case SlakosType::THREEOB:
                    settings::QMSettings::setSlakosType(
                        SlakosType::THREEOB,
                        _resolveBuiltInSlakosPath
                    );
                    settings::QMSettings::setHubbardDerivs(hubbardDerivMap3ob);
                    references::ReferencesOutput::addReferenceFile(
                        references::THREEOB_FILE
                    );
                    break;
                case SlakosType::MATSCI:
                    settings::QMSettings::setSlakosType(
                        SlakosType::MATSCI,
                        _resolveBuiltInSlakosPath
                    );
                    references::ReferencesOutput::addReferenceFile(
                        references::MATSCI_FILE
                    );
                    break;
                case SlakosType::CUSTOM:
                    settings::QMSettings::setSlakosType(SlakosType::CUSTOM);
                    break;
                case SlakosType::NONE: break;
            }
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SlakosType>{
                .metadata   = metaData,
                .notAllowed = {SlakosType::NONE},
                .onSet      = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addSlakosPathKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "slakos_path",
            .title       = "SLAKOS Path",
            .description = "Specifies the path to the SLAKOS executable.",
        };

        const auto setValue = [](const std::string &value)
        { settings::QMSettings::setSlakosPath(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::string>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addThirdOrderKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "third_order",
            .title       = "Third Order DFTB",
            .description = "Specifies whether third order DFTB is used.",
        };

        const auto setValue = [](bool value)
        {
            settings::QMSettings::setUseThirdOrderDftb(value);
            settings::QMSettings::setIsThirdOrderDftbSet(true);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addHubbardDerivsKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "hubbard_derivs",
            .title = "Hubbard Derivatives",
            .description =
                "Specifies custom Hubbard Derivatives for the system.",
        };

        const auto setValue =
            [](const std::unordered_map<std::string, double> &value)
        {
            settings::QMSettings::setHubbardDerivs(value);
            settings::QMSettings::setIsHubbardDerivsSet(true);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::unordered_map<std::string, double>>{
                .metadata = metaData,
                .isArray  = true,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addXtbMethodKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "xtb_method",
            .title       = "xTB Method",
            .description = "Specifies the xTB method to be used.",
        };

        const auto setValue = [](XtbMethod value)
        { settings::QMSettings::setXtbMethod(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<XtbMethod>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addFennolModelPathKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "fennol_model_path",
            .title       = "FeNNol Model Path",
            .description = "Specifies the path to the FeNNol model file.",
        };

        const auto setValue = [](const std::string &value)
        { settings::QMSettings::setFennolModelPath(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::string>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void QMInputParser::addGPUPreprocessingKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "gpu_preprocessing",
            .title = "GPU Preprocessing",
            .description =
                "Specifies whether GPU pre-processing is enabled for FeNNol.",
        };

        const auto setValue = [](bool value)
        { settings::QMSettings::setUseGPUPreprocessing(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief parses the QM method if it starts with "mace"
     *
     * @param model
     *
     * @throws exc::InputFileException if the model is not recognized
     */
    void QMInputParser::parseMaceQMMethod(const std::string &model)
    {
        const auto modelTypeOpt = MaceModelTypeMeta::from_stringCaseInsensitive(
            utilities::toLowerAndReplaceDashesCopy(model)
        );

        if (!modelTypeOpt.has_value())
        {
            const auto allowedValues = mstd::join(
                MaceModelTypeMeta::spellingNames(),
                ", ",
                [](auto &&value) { return utilities::toLowerCopy(value); }
            );

            throw exc::InputFileException(
                std::format(
                    "Invalid mace type qm_method \"{}\" in input file.\n"
                    "Possible values are: {}",
                    model,
                    allowedValues
                )
            );
        }

        switch (modelTypeOpt.value())
        {
            case MaceModelType::MACE_MP:
                references::ReferencesOutput::addReferenceFile(
                    references::MACEMP_FILE
                );
                break;
            case MaceModelType::MACE_OFF:
                references::ReferencesOutput::addReferenceFile(
                    references::MACEOFF_FILE
                );
                break;
            case MaceModelType::MACE_ANICC:
                throw exc::InputFileException(
                    std::format(
                        "The mace ani model is not supported in this version "
                        "of PQ.\n"
                    )
                );
                break;
        }

        settings::QMSettings::setMaceModelType(modelTypeOpt.value());
    }
}   // namespace input
