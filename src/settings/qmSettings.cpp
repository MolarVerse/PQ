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

#include <filesystem>
#include <format>   // for std::format

#include "exceptions.hpp"        // for customException
#include "executablePath.hpp"
#include "stringUtilities.hpp"   // for toLowerCopy

using settings::MaceMode;
using settings::MaceModel;
using settings::MaceModelType;
using settings::QMMethod;
using settings::QMSettings;
using settings::SlakosType;
using settings::XtbMethod;
using namespace customException;
using namespace utilities;

/**
 * @brief returns the qmMethod as string
 *
 * @param method
 * @return std::string
 */
std::string settings::string(const QMMethod method)
{
    switch (method)
    {
        using enum QMMethod;

        case DFTBPLUS: return "DFTBPLUS";
        case ASEDFTBPLUS: return "ASEDFTBPLUS";
        case ASEXTB: return "ASEXTB";
        case PYSCF: return "PYSCF";
        case TURBOMOLE: return "TURBOMOLE";
        case MACE: return "MACE";
        case FENNOL: return "FeNNol";

        case NONE: break;
    }

    return "none";
}

/**
 * @brief returns the maceModel size as string
 *
 * @param model
 * @return std::string
 */
std::string settings::string(const MaceModel model)
{
    switch (model)
    {
        using enum MaceModel;

        case SMALL: return "small";
        case MEDIUM: return "medium";
        case LARGE: return "large";
        case SMALL0B: return "small-0b";
        case MEDIUM0B: return "medium-0b";
        case SMALL0B2: return "small-0b2";
        case MEDIUM0B2: return "medium-0b2";
        case LARGE0B2: return "large-0b2";
        case MEDIUM0B3: return "medium-0b3";
        case MEDIUMMPA0: return "medium-mpa-0";
        case MEDIUMOMAT0: return "medium-omat-0";
        case CUSTOM: return "custom";
    }

    return "none";
}

/**
 * @brief returns the maceModel type as string
 *
 * @param model
 * @return std::string
 */
std::string settings::string(const MaceModelType model)
{
    switch (model)
    {
        using enum MaceModelType;

        case MACE_MP: return "mace_mp";
        case MACE_OFF: return "mace_off";
        case MACE_ANICC: return "mace_anicc";
    }

    return "none";
}

/**
 * @brief returns the maceMode as string
 *
 * @param mode
 * @return std::string
 */
std::string settings::string(const MaceMode mode)
{
    switch (mode)
    {
        using enum MaceMode;

        case ACCURATE: return "accurate";
        case FAST: return "fast";
    }

    return "unknown mode";
}

/**
 * @brief returns the Slakos Type as string
 *
 * @param slakos
 * @return std::string
 */
std::string settings::string(const SlakosType slakos)
{
    switch (slakos)
    {
        using enum SlakosType;

        case THREEOB: return "3ob";
        case MATSCI: return "matsci";
        case CUSTOM: return "custom";

        case NONE: break;
    }

    return "none";
}

/**
 * @brief builds the file path for a built-in SLAKOS set (3ob/matsci)
 *
 * @details installed data next to the executable is preferred. The fetched
 * build-tree data is used while running an uninstalled ASE build.
 */
static std::string builtinSlakosPath([[maybe_unused]] const SlakosType type)
{
#ifdef __SLAKOS_DIR__
    const auto installedPath = utilities::installedDataPath(
        std::filesystem::path("slakos") / settings::string(type) / "skfiles"
    );
    if (std::filesystem::is_directory(installedPath))
        return installedPath.string() +
               std::filesystem::path::preferred_separator;

    const auto buildPath = std::filesystem::path(__SLAKOS_DIR__) /
                           settings::string(type) / "skfiles";
    return buildPath.string() + std::filesystem::path::preferred_separator;
#else
    throw InputFileException(
        "Built-in SLAKOS sets (3ob/matsci) require building PQ with "
        "-DBUILD_WITH_ASE=On"
    );
#endif
}

/**
 * @brief returns the xTB method as string
 *
 * @param method
 * @return std::string
 */
std::string settings::string(const XtbMethod method)
{
    switch (method)
    {
        using enum XtbMethod;

        case GFN1: return "GFN1-xTB";
        case GFN2: return "GFN2-xTB";
        case IPEA1: return "IPEA1-xTB";
    }

    return "none";
}

/**
 * @brief returns an unordered map as string
 *
 * @param unordered_map
 * @return std::string
 */
std::string settings::string(
    const std::unordered_map<std::string, double> unordered_map
)
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
 * @brief returns if the external qm runner is activated
 *
 * @return bool
 */
bool QMSettings::isExternalQMRunner()
{
    using enum QMMethod;

    auto isExternal = false;

    isExternal = isExternal || _qmMethod == DFTBPLUS;
    isExternal = isExternal || _qmMethod == PYSCF;
    isExternal = isExternal || _qmMethod == TURBOMOLE;

    return isExternal;
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
void QMSettings::setQMMethod(const std::string_view &method)
{
    using enum QMMethod;
    const auto methodToLowerAndReplaceDashes =
        toLowerAndReplaceDashesCopy(method);

    if ("dftbplus" == methodToLowerAndReplaceDashes)
        _qmMethod = DFTBPLUS;

    else if ("pyscf" == methodToLowerAndReplaceDashes)
        _qmMethod = PYSCF;

    else if ("turbomole" == methodToLowerAndReplaceDashes)
        _qmMethod = TURBOMOLE;

    else if ("mace" == methodToLowerAndReplaceDashes)
        _qmMethod = MACE;

    else if ("ase_dftbplus" == methodToLowerAndReplaceDashes)
        _qmMethod = ASEDFTBPLUS;

    else if ("ase_xtb" == methodToLowerAndReplaceDashes)
        _qmMethod = ASEXTB;

    else if ("fennol" == methodToLowerAndReplaceDashes)
        _qmMethod = FENNOL;

    else
        _qmMethod = NONE;
}

/**
 * @brief sets the qmMethod to enum in settings
 *
 * @param method
 */
void QMSettings::setQMMethod(const QMMethod method) { _qmMethod = method; }

/**
 * @brief sets the maceModel to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModel(const std::string_view &model)
{
    using enum MaceModel;
    const auto modelToLowerAndReplaceDashes =
        toLowerAndReplaceDashesCopy(model);

    if ("small" == modelToLowerAndReplaceDashes)
        _maceModel = SMALL;

    else if ("medium" == modelToLowerAndReplaceDashes)
        _maceModel = MEDIUM;

    else if ("large" == modelToLowerAndReplaceDashes)
        _maceModel = LARGE;

    else if ("small_0b" == modelToLowerAndReplaceDashes)
        _maceModel = SMALL0B;

    else if ("medium_0b" == modelToLowerAndReplaceDashes)
        _maceModel = MEDIUM0B;

    else if ("small_0b2" == modelToLowerAndReplaceDashes)
        _maceModel = SMALL0B2;

    else if ("medium_0b2" == modelToLowerAndReplaceDashes)
        _maceModel = MEDIUM0B2;

    else if ("large_0b2" == modelToLowerAndReplaceDashes)
        _maceModel = LARGE0B2;

    else if ("medium_0b3" == modelToLowerAndReplaceDashes)
        _maceModel = MEDIUM0B3;

    else if ("medium_mpa_0" == modelToLowerAndReplaceDashes)
        _maceModel = MEDIUMMPA0;

    else if ("medium_omat_0" == modelToLowerAndReplaceDashes)
        _maceModel = MEDIUMOMAT0;

    else if ("custom" == modelToLowerAndReplaceDashes)
        _maceModel = CUSTOM;

    else
        throw UserInputException(
            std::format("Mace model size {} not recognized", model)
        );
}

/**
 * @brief sets the maceModel to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModel(const MaceModel model) { _maceModel = model; }

/**
 * @brief sets the maceModelType to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModelType(const std::string_view &model)
{
    using enum MaceModelType;
    const auto modelToLower = toLowerAndReplaceDashesCopy(model);

    if ("mace_mp" == modelToLower)
        _maceModelType = MACE_MP;

    else if ("mace_off" == modelToLower)
        _maceModelType = MACE_OFF;

    else if ("mace_anicc" == modelToLower)
        _maceModelType = MACE_ANICC;

    else
        throw UserInputException(
            std::format("Mace {} model not recognized", model)
        );
}

/**
 * @brief sets the maceModelType to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModelType(const MaceModelType model)
{
    _maceModelType = model;
}

/**
 * @brief sets the maceMode to enum in settings
 *
 * @param mode
 */
void QMSettings::setMaceMode(const std::string_view &mode)
{
    using enum MaceMode;
    const auto modeToLower = toLowerAndReplaceDashesCopy(mode);

    if ("accurate" == modeToLower)
        _maceMode = ACCURATE;

    else if ("fast" == modeToLower)
        _maceMode = FAST;

    else
        throw UserInputException(std::format(
            "Unknown mace_mode \"{}\". Valid values are \"accurate\" (exact "
            "e3nn reference) or \"fast\" (cuequivariance-accelerated).",
            mode
        ));
}

/**
 * @brief sets the maceMode to enum in settings
 *
 * @param mode
 */
void QMSettings::setMaceMode(const MaceMode mode) { _maceMode = mode; }

/**
 * @brief set the mace model path
 *
 */
void QMSettings::setMaceModelPath(const std::string_view &path)
{
    _maceModelPath = path;
}

/**
 * @brief sets the XtbMethod to enum in settings
 *
 * @param method
 */
void QMSettings::setXtbMethod(const std::string_view &method)
{
    using enum XtbMethod;
    const auto xtbMethod = toLowerAndReplaceDashesCopy(method);

    if ("gfn1_xtb" == xtbMethod)
        _xtbMethod = GFN1;

    else if ("gfn2_xtb" == xtbMethod)
        _xtbMethod = GFN2;

    else if ("ipea1_xtb" == xtbMethod)
        _xtbMethod = IPEA1;

    else
        throw UserInputException(
            std::format("xTB method \"{}\" not recognized", method)
        );
}

/**
 * @brief sets the xTB method to enum in settings
 *
 * @param method
 */
void QMSettings::setXtbMethod(const XtbMethod method) { _xtbMethod = method; }

/**
 * @brief sets the qmScript in settings
 *
 * @param script
 */
void QMSettings::setQMScript(const std::string_view &script)
{
    _qmScript = script;
}

/**
 * @brief sets the qmScriptFullPath in settings
 *
 * @param script
 */
void QMSettings::setQMScriptFullPath(const std::string_view &script)
{
    _qmScriptFullPath = script;
}

/**
 * @brief sets the slakosType to enum in settings
 *
 * @param slakos
 */
void QMSettings::setSlakosType(const std::string_view &slakos)
{
    using enum SlakosType;
    const auto slakosType = toLowerAndReplaceDashesCopy(slakos);

    if ("3ob" == slakosType)
    {
        _slakosType = THREEOB;
        _slakosPath = builtinSlakosPath(_slakosType);
    }

    else if ("matsci" == slakosType)
    {
        _slakosType = MATSCI;
        _slakosPath = builtinSlakosPath(_slakosType);
    }

    else if ("custom" == slakosType)
        _slakosType = CUSTOM;

    else if ("none" == slakosType)
    {
        _slakosType = NONE;
        _slakosPath = "";
    }

    else
        throw UserInputException(
            std::format("Slakos {} not recognized", slakos)
        );
}

/**
 * @brief sets the slakosType to enum in settings
 *
 * @param slakos
 * @param resolveBuiltInPath
 */
void QMSettings::setSlakosType(
    const SlakosType slakos,
    const bool       resolveBuiltInPath
)
{
    if (!resolveBuiltInPath &&
        (slakos == SlakosType::THREEOB || slakos == SlakosType::MATSCI))
    {
        _slakosType = slakos;
        _slakosPath.clear();
        return;
    }

    setSlakosType(string(slakos));
}

/**
 * @brief sets the slakosPath in settings
 *
 * @param path
 */
void QMSettings::setSlakosPath(const std::string_view &path)
{
    if (_slakosType == SlakosType::CUSTOM)
        _slakosPath = path;

    else if (_slakosType == SlakosType::NONE)
        throw UserInputException(
            "Slakos path cannot be set without a slakos type"
        );

    else
    {
        throw UserInputException(
            std::format(
                "Slakos path cannot be set for slakos type: {}",
                string(_slakosType)
            )
        );
    }
}

/**
 * @brief sets if third order DFTB should be used
 *
 */
void QMSettings::setUseThirdOrderDftb(const bool useThirdOrderDftb)
{
    _useThirdOrderDftb = useThirdOrderDftb;
}

/**
 * @brief sets if the third order is set
 *
 */
void QMSettings::setIsThirdOrderDftbSet(const bool isThirdOrderDftbSet)
{
    _isThirdOrderDftbSet = isThirdOrderDftbSet;
}

/**
 * @brief sets the custom Hubbard Derivative dictionary
 *
 */
void QMSettings::setHubbardDerivs(
    std::unordered_map<std::string, double> hubbardDerivs
)
{
    _hubbardDerivs = hubbardDerivs;
}

/**
 * @brief sets if the Hubbard Derivative dictionary is set by the user
 *
 */
void QMSettings::setIsHubbardDerivsSet(const bool isHubbardDerivsSet)
{
    _isHubbardDerivsSet = isHubbardDerivsSet;
}

/**
 * @brief sets if the dispersion correction should be used
 *
 */
void QMSettings::setUseDispersionCorrection(const bool useDispersionCorr)
{
    _useDispersionCorrection = useDispersionCorr;
}

/**
 * @brief sets if the net force should be removed after reading in the QM forces
 *
 */
void QMSettings::setRemoveNetForce(const bool removeNetForce)
{
    _removeNetForce = removeNetForce;
}

/**
 * @brief sets the qmLoopTimeLimit in settings
 *
 * @param time
 */
void QMSettings::setQMLoopTimeLimit(const double time)
{
    _qmLoopTimeLimit = time;
}

/**
 * @brief sets the FeNNol model path
 *
 * @param script
 */
void QMSettings::setFennolModelPath(const std::string_view &path)
{
    _fennolModelPath = path;
}

/**
 * @brief sets if the GPU pre-processing should be enabled for FeNNol
 *
 */
void QMSettings::setUseGPUPreprocessing(const bool useGPUPreprocessing)
{
    _useGPUPreprocessing = useGPUPreprocessing;
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
QMMethod QMSettings::getQMMethod() { return _qmMethod; }

/**
 * @brief returns the maceModel
 *
 * @return MaceModel
 */
MaceModel QMSettings::getMaceModel() { return _maceModel; }

MaceModelType QMSettings::getMaceModelType() { return _maceModelType; }

/**
 * @brief returns the maceMode
 *
 * @return MaceMode
 */
MaceMode QMSettings::getMaceMode() { return _maceMode; }

/**
 * @brief returns the maceModelPath
 *
 * @return std::string
 */
std::string QMSettings::getMaceModelPath() { return _maceModelPath; }

/**
 * @brief returns the qmScript
 *
 * @return std::string
 */
std::string QMSettings::getQMScript() { return _qmScript; }

/**
 * @brief returns the qmScriptFullPath
 *
 * @return std::string
 */
std::string QMSettings::getQMScriptFullPath() { return _qmScriptFullPath; }

/**
 * @brief returns the slakosType
 *
 * @return SlakosType
 */
SlakosType QMSettings::getSlakosType() { return _slakosType; }

/**
 * @brief returns the slakosPath
 *
 * @return std::string
 */
std::string QMSettings::getSlakosPath() { return _slakosPath; }

/**
 * @brief returns if third order DFTB should be used
 *
 * @return bool
 */
bool QMSettings::useThirdOrderDftb() { return _useThirdOrderDftb; }

/**
 * @brief returns if the third order is set
 *
 * @return bool
 */
bool QMSettings::isThirdOrderDftbSet() { return _isThirdOrderDftbSet; }

/**
 * @brief returns if the Hubbard derivatives are set by the user
 *
 * @return bool
 */
bool QMSettings::isHubbardDerivsSet() { return _isHubbardDerivsSet; }

/**
 * @brief returns the Hubbard Derivative dictionary
 *
 * @return std::unordered_map<std::string, double>
 */
std::unordered_map<std::string, double> QMSettings::getHubbardDerivs()
{
    return _hubbardDerivs;
}

/**
 * @brief returns if the dispersion correction should be used
 *
 * @return bool
 */
bool QMSettings::useDispersionCorr() { return _useDispersionCorrection; }

/**
 * @brief returns if the net force should be removed after reading in the QM
 * forces
 *
 * @return bool
 */
bool QMSettings::getRemoveNetForce() { return _removeNetForce; }

/**
 * @brief returns the xTBMethod
 *
 * @return XtbMethod
 */
XtbMethod QMSettings::getXtbMethod() { return _xtbMethod; }

/**
 * @brief returns the qmLoopTimeLimit
 *
 * @return double
 */
double QMSettings::getQMLoopTimeLimit() { return _qmLoopTimeLimit; }

/**
 * @brief returns the FeNNol model path
 *
 * @return std::string
 */
std::string QMSettings::getFennolModelPath() { return _fennolModelPath; }

/**
 * @brief returns if GPU pre-processing should be used for FeNNol
 *
 * @return bool
 */
bool QMSettings::useGPUPreprocessing() { return _useGPUPreprocessing; }
