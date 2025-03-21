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

#include <format>   // for std::format

#include "exceptions.hpp"        // for customException
#include "stringUtilities.hpp"   // for toLowerCopy

using settings::MaceModelSize;
using settings::MaceModelType;
using settings::QMMethod;
using settings::QMSettings;
using settings::SlakosType;
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
        case PYSCF: return "PYSCF";
        case TURBOMOLE: return "TURBOMOLE";
        case MACE: return "MACE";

        default: return "none";
    }
}

/**
 * @brief returns the maceModel size as string
 *
 * @param model
 * @return std::string
 */
std::string settings::string(const MaceModelSize model)
{
    switch (model)
    {
        using enum MaceModelSize;

        case LARGE: return "large";
        case MEDIUM: return "medium";
        case SMALL: return "small";
        case SMALL0B: return "small-0b";

        default: return "none";
    }
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

        default: return "none";
    }
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

        default: return "none";
    }
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
void QMSettings::setMaceModelSize(const std::string_view &model)
{
    using enum MaceModelSize;
    const auto modelToLower = toLowerCopy(model);

    if ("large" == modelToLower)
        _maceModelSize = LARGE;

    else if ("medium" == modelToLower)
        _maceModelSize = MEDIUM;

    else if ("small" == modelToLower)
        _maceModelSize = SMALL;

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
void QMSettings::setMaceModelSize(const MaceModelSize model)
{
    _maceModelSize = model;
}

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
 * @brief set the mace model path
 *
 */
void QMSettings::setMaceModelPath(const std::string_view &path)
{
    _maceModelPath = path;
}

/**
 * @brief sets the qmScript in settings
 *
 * @param script
 */
void QMSettings::setQMScript(const std::string_view &script)
{
    _qmScript = toLowerAndReplaceDashesCopy(script);
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
        _slakosPath = __SLAKOS_DIR__ + string(_slakosType) + "/skfiles/";
    }

    else if ("matsci" == slakosType)
    {
        _slakosType = MATSCI;
        _slakosPath = __SLAKOS_DIR__ + string(_slakosType) + "/skfiles/";
    }

    else if ("custom" == slakosType)
        _slakosType = CUSTOM;

    else if ("none" == slakosType)
    {
        _slakosType = NONE;
        _slakosPath = "";
    }

    else
        throw UserInputException(std::format("Slakos {} not recognized", slakos)
        );
}

/**
 * @brief sets the slakosType to enum in settings
 *
 * @param model
 */
void QMSettings::setSlakosType(const SlakosType slakos)
{
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
        throw UserInputException(std::format(
            "Slakos path cannot be set for slakos type: {}",
            string(_slakosType)
        ));
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
 * @brief sets the qmLoopTimeLimit in settings
 *
 * @param time
 */
void QMSettings::setQMLoopTimeLimit(const double time)
{
    _qmLoopTimeLimit = time;
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
 * @return MaceModelSize
 */
MaceModelSize QMSettings::getMaceModelSize() { return _maceModelSize; }

MaceModelType QMSettings::getMaceModelType() { return _maceModelType; }

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
 * @brief returns the qmLoopTimeLimit
 *
 * @return double
 */
double QMSettings::getQMLoopTimeLimit() { return _qmLoopTimeLimit; }