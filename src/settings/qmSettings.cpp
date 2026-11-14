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

#include "qmSettings.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <optional>
#include <utility>

#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "executablePath.hpp"
#include "hubbardDerivMap.hpp"
#include "referenceSettings.hpp"
#include "stringUtilities.hpp"

#define SCRIPT_PATH_  _SCRIPT_PATH_
#define SINGULARITY_  _SINGULARITY_
#define STATIC_BUILD_ _STATIC_BUILD_

namespace
{
    /**
     * @brief builds the file path for a built-in SLAKOS set (3ob/matsci)
     *
     * @details installed data next to the executable is preferred. The
     * fetched build-tree data is used while running an uninstalled ASE
     * build.
     */
    std::string builtinSlakosPath([[maybe_unused]] const SlakosType type)
    {
#ifdef __SLAKOS_DIR__
        const auto installedPath = utilities::installedDataPath(
            std::filesystem::path("slakos") / SlakosTypeMeta::toString(type) /
            "skfiles"
        );
        if (std::filesystem::is_directory(installedPath))
            return installedPath.string() +
                   std::filesystem::path::preferred_separator;

        const auto buildPath = std::filesystem::path(__SLAKOS_DIR__) /
                               SlakosTypeMeta::toString(type) / "skfiles";
        return buildPath.string() + std::filesystem::path::preferred_separator;
#else
        throw exc::InputFileException(
            "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
            "-DBUILD_WITH_ASE=On"
        );
#endif
    }
}   // namespace

/**
 * @brief returns an unordered map as string
 *
 * @param unordered_map
 * @return std::string
 */
std::string string(const std::unordered_map<std::string, double> &unordered_map)
{
    std::string unorderedMapStr;
    for (const auto &pair : unordered_map)
    {
        if (!unorderedMapStr.empty())
            unorderedMapStr += ", ";
        unorderedMapStr += std::format("{}: {}", pair.first, pair.second);
    }

    return unorderedMapStr;
}

/**
 * @brief finalizes the QM settings by performing necessary checks and setups
 *
 */
void QMSettings::finalize() {}

/**
 * @brief finalizes the DFTB settings by performing necessary checks and setups
 *
 */
void DftbSettings::finalize(ReferenceSettings &referenceSettings)
{
    switch (_slakosType.get())
    {
        case SlakosType::THREEOB:
            if (!_useThirdOrderDftb.isSetByUser())
                _useThirdOrderDftb.setBySystem(true);
            if (!_hubbardDerivs.isSetByUser())
                _hubbardDerivs.setBySystem(hubbardDerivMap3ob);
            referenceSettings.addReferenceFile(ReferenceFiles::THREEOB_FILE);
            break;
        case SlakosType::MATSCI:
            referenceSettings.addReferenceFile(ReferenceFiles::MATSCI_FILE);
            break;
        case SlakosType::CUSTOM:
        case SlakosType::NONE: break;
    }
}

/**
 * @brief finalizes the XTB settings by performing necessary checks and setups
 *
 */
void XtbSettings::finalize(ReferenceSettings &referenceSettings)
{
    switch (_xtbMethod.get())
    {
        case XtbMethod::GFN1:
            referenceSettings.addReferenceFile(ReferenceFiles::GFN1_FILE);
            break;
        case XtbMethod::GFN2:
            referenceSettings.addReferenceFile(ReferenceFiles::GFN2_FILE);
            break;
        case XtbMethod::IPEA1:
            referenceSettings.addReferenceFile(ReferenceFiles::IPEA1_FILE);
            break;
    }
}

namespace
{
    /**
     * @brief returns the full path to the bundled QM script
     *
     * @param script The name of the QM script
     * @return std::string The full path to the bundled QM script
     */
    std::string bundledQMScriptPath(const std::string_view script)
    {
        const auto installedPath = utilities::installedDataPath(
            std::filesystem::path("scripts") / script
        );
        if (std::filesystem::is_regular_file(installedPath))
            return installedPath.string();

        return (std::filesystem::path(SCRIPT_PATH_) / script).string();
    }
}   // namespace

/**
 * @brief finalizes the external QM settings by performing necessary checks and
 *setups
 *
 * @details if a singularity or static build is used the qm_script is set to
 * the qm_script_full_path and the script path is set to the empty string to
 * avoid errors. This is necessary because the script can not be accessed
 * from inside the container. Therefore the user has to provide the script
 * somewhere else and give the full or relative path to it. For more
 * information please refer to the documentation.
 */
void ExternalQMSettings::finalize()
{
    const std::string singularityString = SINGULARITY_;
    const std::string staticBuildString = STATIC_BUILD_;

    const auto singularity = utilities::toLowerCopy(singularityString) == "on";
    const auto staticBuild = utilities::toLowerCopy(staticBuildString) == "on";

    if (singularity || staticBuild)
    {
        if (!_qmScriptFullPath.isSetByUser())
        {
            throw exc::QMRunnerException(
                "You are using at least one of these settings: i) "
                "singularity build or/and ii) static build of PQ. Therefore "
                "the general setting with 'qm_script' to set only the name of "
                "the executable is not applicable. Please use "
                "'qm_script_full_path' instead and provide the full path "
                "to the executable. For singularity builds the script can not "
                "be accessed from inside the container. In case of a static "
                "build the binary may be shipped without the source code and "
                "again PQ might therefore not be able to locate the executable "
                "qm script. Therefore you have to provide the script "
                "somewhere else and give the full/relative path to it. For "
                "more information please refer to the documentation."
            );
        }

        if (_qmScript.isSetByUser())
        {
            throw exc::QMRunnerException(
                "You have set both 'qm_script' and 'qm_script_full_path' "
                "in the input file. Please use only one the full path option "
                "as you are working either with a singularity build or a "
                "static build. For more information please refer to the "
                "documentation."
            );
        }
        // overwriting qm_script with full path
        _qmScript.setBySystem(_qmScriptFullPath.get());
    }
    else if (!_qmScript.isSetByUser() && !_qmScriptFullPath.isSetByUser())
    {
        throw exc::InputFileException(
            "No qm_script provided. Please provide a qm_script in the "
            "input "
            "file."
        );
    }
    else if (_qmScriptFullPath.isSetByUser() && !_qmScript.isSetByUser())
    {
        // overwriting qm_script with full path
        _qmScript.setBySystem(_qmScriptFullPath.get());
    }
    else if (_qmScriptFullPath.isSetByUser() && _qmScript.isSetByUser())
    {
        throw exc::InputFileException(
            "You have set both 'qm_script' and 'qm_script_full_path' in "
            "the input file. They are mutually exclusive. Please use only one "
            "of them. For more information please refer to the documentation."
        );
    }
    else
    {
        _qmScript.setBySystem(bundledQMScriptPath(_qmScript.get()));
    }
}

/**
 * @brief returns if the external qm runner is activated
 *
 * @return bool
 */
bool QMSettings::isExternalQMRunner() const
{
    switch (_qmMethod.get())
    {
        case QMMethod::DFTBPLUS:
        case QMMethod::PYSCF:
        case QMMethod::TURBOMOLE:
        case QMMethod::ORCA: return true;

        case QMMethod::NONE:
        case QMMethod::ASE_XTB:
        case QMMethod::MACE:
        case QMMethod::FENNOL:
        case QMMethod::ASE_DFTBPLUS: return false;
    }

    std::unreachable();
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief sets the qmMethod to enum in settings
 *
 * @param method
 */
void QMSettings::setQMMethodByUser(QMMethod method)
{
    _qmMethod.setByUser(method);
}

/**
 * @brief sets the maceModel to enum in settings
 *
 * @param model
 */
void MaceSettings::setMaceModelByUser(MaceModel model)
{
    _maceModel.setByUser(model);
}

/**
 * @brief sets the maceModelType to enum in settings
 *
 * @param model
 */
void MaceSettings::setMaceModelTypeByUser(MaceModelType model)
{
    _maceModelType.setByUser(model);
}

/**
 * @brief sets the maceMode to enum in settings
 *
 * @param mode
 */
void MaceSettings::setMaceModeByUser(MaceMode mode)
{
    _maceMode.setByUser(mode);
}

/**
 * @brief set the mace model path
 *
 */
void MaceSettings::setMaceModelPathByUser(const std::string &path)
{
    _maceModelPath.setByUser(path);
}

/**
 * @brief sets the xTB method to enum in settings
 *
 * @param method
 */
void XtbSettings::setXtbMethodByUser(XtbMethod method)
{
    _xtbMethod.setByUser(method);
}

/**
 * @brief sets the qmScript in settings
 *
 * @param script
 */
void ExternalQMSettings::setQMScriptByUser(const std::string &script)
{
    _qmScript.setByUser(script);
}

/**
 * @brief sets the qmScriptFullPath in settings
 *
 * @param scriptFullPath
 */
void ExternalQMSettings::setQMScriptFullPathByUser(
    const std::string &scriptFullPath
)
{
    _qmScriptFullPath.setByUser(scriptFullPath);
}

/**
 * @brief sets the slakosType to enum in settings
 *
 * @param slakos
 */
void DftbSettings::setSlakosTypeByUser(SlakosType slakos)
{
    setSlakosTypeByUser(slakos, true);
}

/**
 * @brief sets the slakosType to enum in settings
 *
 * @param slakos
 * @param resolveBuiltInPath
 */
void DftbSettings::setSlakosTypeByUser(
    SlakosType slakos,
    bool       resolveBuiltInPath
)
{
    if (!resolveBuiltInPath &&
        (slakos == SlakosType::THREEOB || slakos == SlakosType::MATSCI))
    {
        _slakosType.setByUser(slakos);
        _slakosPath.setByUser("");
        return;
    }

    switch (slakos)
    {
        using enum SlakosType;

        case THREEOB:
        case MATSCI:
            if (resolveBuiltInPath)
                _slakosPath.setBySystem(builtinSlakosPath(slakos));
            break;
        case CUSTOM:
        case NONE: _slakosPath.setByUser(""); break;
    }

    _slakosType.setByUser(slakos);
}

/**
 * @brief sets the slakosPath in settings
 *
 * @param path
 */
void DftbSettings::setSlakosPathByUser(const std::string &path)
{
    if (_slakosType.get() == SlakosType::CUSTOM)
        _slakosPath.setByUser(path);
    else if (_slakosType.get() == SlakosType::NONE)
    {
        throw exc::UserInputException(
            "Slakos path cannot be set without a slakos type"
        );
    }
    else
    {
        throw exc::UserInputException(
            std::format(
                "Slakos path cannot be set for slakos type: {}",
                SlakosTypeMeta::toString(_slakosType.get())
            )
        );
    }
}

/**
 * @brief sets if third order DFTB should be used
 *
 */
void DftbSettings::setUseThirdOrderDftbByUser(bool useThirdOrderDftb)
{
    _useThirdOrderDftb.setByUser(useThirdOrderDftb);
}

/**
 * @brief sets the custom Hubbard Derivative dictionary
 *
 */
void DftbSettings::setHubbardDerivsByUser(
    const std::unordered_map<std::string, double> &hubbardDerivs
)
{
    _hubbardDerivs.setByUser(hubbardDerivs);
}

/**
 * @brief sets if the dispersion correction should be used
 *
 */
void QMSettings::setUseDispersionCorrectionByUser(bool useDispersionCorr)
{
    _useDispersionCorrection.setByUser(useDispersionCorr);
}

/**
 * @brief sets if the net force should be removed after reading in the QM
 * forces
 *
 */
void QMSettings::setRemoveNetForceByUser(bool removeNetForce)
{
    _removeNetForce.setByUser(removeNetForce);
}

/**
 * @brief sets the qmLoopTimeLimit in settings
 *
 * @param time
 */
void QMSettings::setQMLoopTimeLimitByUser(const TimeMs &time)
{
    _qmLoopTimeLimit.setByUser(time);
}

/**
 * @brief sets the FeNNol model path
 *
 * @param path
 */
void FennolSettings::setFennolModelPathByUser(const std::string &path)
{
    _fennolModelPath.setByUser(path);
}

/**
 * @brief sets if the GPU pre-processing should be enabled for FeNNol
 *
 */
void FennolSettings::setUseGPUPreprocessingByUser(bool useGPUPreprocessing)
{
    _useGPUPreprocessing.setByUser(useGPUPreprocessing);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief returns the qmMethod
 *
 * @return QMMethod
 */
QMMethod QMSettings::getQMMethod() const { return _qmMethod.get(); }

/**
 * @brief returns if the QM method was set by the user
 *
 * @return bool
 */
bool QMSettings::isQMMethodSetByUser() const { return _qmMethod.isSetByUser(); }

/**
 * @brief returns the maceModel
 *
 * @return MaceModel
 */
MaceModel MaceSettings::getMaceModel() const { return _maceModel.get(); }

/**
 * @brief returns the maceModelType
 *
 * @return MaceModelType
 */
MaceModelType MaceSettings::getMaceModelType() const
{
    return _maceModelType.get();
}

/**
 * @brief returns the maceModel as expected by mace.calculators
 *
 * @return std::string e.g. "small-0b"
 */
std::string MaceSettings::getMaceModelName() const
{
    auto name =
        utilities::toLowerCopy(MaceModelMeta::toString(_maceModel.get()));

    std::ranges::replace(name, '_', '-');

    return name;
}

/**
 * @brief returns the maceModelType as calculator name in mace.calculators
 *
 * @return std::string e.g. "mace_off"
 */
std::string MaceSettings::getMaceModelTypeName() const
{
    // unfortunately, when using the testing mode, we need to return the
    // specific calculator name that is used for testing purposes, which is
    // "MACECalculator". Otherwise, we need to install mace-torch for all simple
    // unit tests
    if (_useForTesting)
        return _testModelTypeStr;

    return utilities::toLowerCopy(
        MaceModelTypeMeta::toString(_maceModelType.get())
    );
}

/**
 * @brief returns the maceMode
 *
 * @return MaceMode
 */
MaceMode MaceSettings::getMaceMode() const { return _maceMode.get(); }

/**
 * @brief returns the maceModelPath
 *
 * @return std::string
 */
std::string MaceSettings::getMaceModelPath() const
{
    return _maceModelPath.get();
}

/**
 * @brief returns the qmScript
 *
 * @return const std::string&
 */
const std::string &ExternalQMSettings::getQMScript() const
{
    return _qmScript.get();
}

/**
 * @brief returns the qmScriptFullPath
 *
 * @return const std::string&
 */
const std::string &ExternalQMSettings::getQMScriptFullPath() const
{
    return _qmScriptFullPath.get();
}

/**
 * @brief returns the slakosType
 *
 * @return SlakosType
 */
SlakosType DftbSettings::getSlakosType() const { return _slakosType.get(); }

/**
 * @brief returns the slakosPath
 *
 * @return const std::string&
 */
const std::string &DftbSettings::getSlakosPath() const
{
    return _slakosPath.get();
}

/**
 * @brief returns if third order DFTB should be used
 *
 * @return bool
 */
bool DftbSettings::useThirdOrderDftb() const
{
    return _useThirdOrderDftb.get();
}

/**
 * @brief returns the Hubbard Derivative dictionary
 *
 * @return std::unordered_map<std::string, double>
 */
std::optional<std::unordered_map<std::string, double>> DftbSettings::
    getHubbardDerivs() const
{
    if (!_hubbardDerivs.isSetBySomeone())
        return std::nullopt;

    return _hubbardDerivs.get();
}

/**
 * @brief returns if the Hubbard Derivative dictionary was set by the user
 *
 * @return bool
 */
bool DftbSettings::isHubbardDerivsSetByUser() const
{
    return _hubbardDerivs.isSetByUser();
}

/**
 * @brief returns if the dispersion correction should be used
 *
 * @return bool
 */
bool QMSettings::useDispersionCorr() const
{
    return _useDispersionCorrection.get();
}

/**
 * @brief returns if the net force should be removed after reading in the QM
 * forces
 *
 * @return bool
 */
bool QMSettings::getRemoveNetForce() const { return _removeNetForce.get(); }

/**
 * @brief returns the xTBMethod
 *
 * @return XtbMethod
 */
XtbMethod XtbSettings::getXtbMethod() const { return _xtbMethod.get(); }

/**
 * @brief returns the qmLoopTimeLimit
 *
 * @return const TimeMs&
 */
const TimeMs &QMSettings::getQMLoopTimeLimit() const
{
    return _qmLoopTimeLimit.get();
}

/**
 * @brief returns the FeNNol model path
 *
 * @return const std::string&
 */
const std::string &FennolSettings::getFennolModelPath() const
{
    return _fennolModelPath.get();
}

/**
 * @brief returns if GPU pre-processing should be used for FeNNol
 *
 * @return bool
 */
bool FennolSettings::useGPUPreprocessing() const
{
    return _useGPUPreprocessing.get();
}
