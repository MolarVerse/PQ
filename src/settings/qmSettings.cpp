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
#include <format>
#include <utility>

#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "executablePath.hpp"

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
 * @brief constructs a QMSettings object with default values for its members
 *
 * @details The QM method is initialized to ASE_XTB, the MACE model to MEDIUM,
 * the MACE model type to MACE_MP, and the MACE mode to ACCURATE.
 */
QMSettings::QMSettings()
    : _maceModel(MaceModel::MEDIUM),
      _maceModelType(MaceModelType::MACE_MP),
      _maceMode(MaceMode::ACCURATE),
      _slakosType(SlakosType::NONE),
      _xtbMethod(XtbMethod::GFN2)
{
}

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
 * @brief returns if the external qm runner is activated
 *
 * @return bool
 */
bool QMSettings::isExternalQMRunner() const
{
    using enum QMMethod;

    switch (_qmMethod.get())
    {
        case DFTBPLUS:
        case PYSCF:
        case TURBOMOLE: return true;
        case ASE_DFTBPLUS:
        case ASE_XTB:
        case MACE:
        case FENNOL: return false;
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
void QMSettings::setQMMethod(QMMethod method) { _qmMethod.set(method); }

/**
 * @brief sets the maceModel to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModel(MaceModel model) { _maceModel.set(model); }

/**
 * @brief sets the maceModelType to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModelType(MaceModelType model)
{
    _maceModelType.set(model);
}

/**
 * @brief sets the maceMode to enum in settings
 *
 * @param mode
 */
void QMSettings::setMaceMode(MaceMode mode) { _maceMode.set(mode); }

/**
 * @brief set the mace model path
 *
 */
void QMSettings::setMaceModelPath(const std::string_view &path)
{
    _maceModelPath = path;
}

/**
 * @brief sets the xTB method to enum in settings
 *
 * @param method
 */
void QMSettings::setXtbMethod(XtbMethod method) { _xtbMethod.set(method); }

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
void QMSettings::setSlakosType(SlakosType slakos)
{
    setSlakosType(slakos, true);
}

/**
 * @brief sets the slakosType to enum in settings
 *
 * @param slakos
 * @param resolveBuiltInPath
 */
void QMSettings::setSlakosType(SlakosType slakos, bool resolveBuiltInPath)
{
    if (!resolveBuiltInPath &&
        (slakos == SlakosType::THREEOB || slakos == SlakosType::MATSCI))
    {
        _slakosType.set(slakos);
        _slakosPath.clear();
        return;
    }

    switch (slakos)
    {
        using enum SlakosType;

        case THREEOB:
        case MATSCI:
            if (resolveBuiltInPath)
                _slakosPath = builtinSlakosPath(slakos);
            break;
        case CUSTOM:
        case NONE: _slakosPath.clear(); break;
    }

    _slakosType.set(slakos);
}

/**
 * @brief sets the slakosPath in settings
 *
 * @param path
 */
void QMSettings::setSlakosPath(const std::string_view &path)
{
    if (_slakosType.get() == SlakosType::CUSTOM)
        _slakosPath = path;
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
void QMSettings::setUseThirdOrderDftb(bool useThirdOrderDftb)
{
    _useThirdOrderDftb = useThirdOrderDftb;
}

/**
 * @brief sets if the third order is set
 *
 */
void QMSettings::setIsThirdOrderDftbSet(bool isThirdOrderDftbSet)
{
    _isThirdOrderDftbSet = isThirdOrderDftbSet;
}

/**
 * @brief sets the custom Hubbard Derivative dictionary
 *
 */
void QMSettings::setHubbardDerivs(
    const std::unordered_map<std::string, double> &hubbardDerivs
)
{
    _hubbardDerivs = hubbardDerivs;
}

/**
 * @brief sets if the Hubbard Derivative dictionary is set by the user
 *
 */
void QMSettings::setIsHubbardDerivsSet(bool isHubbardDerivsSet)
{
    _isHubbardDerivsSet = isHubbardDerivsSet;
}

/**
 * @brief sets if the dispersion correction should be used
 *
 */
void QMSettings::setUseDispersionCorrection(bool useDispersionCorr)
{
    _useDispersionCorrection = useDispersionCorr;
}

/**
 * @brief sets if the net force should be removed after reading in the QM
 * forces
 *
 */
void QMSettings::setRemoveNetForce(bool removeNetForce)
{
    _removeNetForce = removeNetForce;
}

/**
 * @brief sets the qmLoopTimeLimit in settings
 *
 * @param time
 */
void QMSettings::setQMLoopTimeLimit(double time) { _qmLoopTimeLimit = time; }

/**
 * @brief sets the FeNNol model path
 *
 * @param path
 */
void QMSettings::setFennolModelPath(const std::string_view &path)
{
    _fennolModelPath = path;
}

/**
 * @brief sets if the GPU pre-processing should be enabled for FeNNol
 *
 */
void QMSettings::setUseGPUPreprocessing(bool useGPUPreprocessing)
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
QMMethod QMSettings::getQMMethod() const { return _qmMethod.get(); }

/**
 * @brief returns if the QM method is set
 *
 * @return bool
 */
bool QMSettings::isQMMethodSet() const { return _qmMethod.isSet(); }

/**
 * @brief returns the maceModel
 *
 * @return MaceModel
 */
MaceModel QMSettings::getMaceModel() const { return _maceModel.get(); }

/**
 * @brief returns the maceModelType
 *
 * @return MaceModelType
 */
MaceModelType QMSettings::getMaceModelType() const
{
    return _maceModelType.get();
}

/**
 * @brief returns the maceMode
 *
 * @return MaceMode
 */
MaceMode QMSettings::getMaceMode() const { return _maceMode.get(); }

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
SlakosType QMSettings::getSlakosType() const { return _slakosType.get(); }

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
XtbMethod QMSettings::getXtbMethod() const { return _xtbMethod.get(); }

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
