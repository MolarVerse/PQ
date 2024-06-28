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
using settings::QMMethod;
using settings::QMSettings;
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
        case PYSCF: return "PYSCF";
        case TURBOMOLE: return "TURBOMOLE";
        case MACE: return "MACE";

        default: return "none";
    }
}

/**
 * @brief returns the maceModel as string
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

        default: return "none";
    }
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
    const auto methodToLower = toLowerCopy(method);

    if ("dftbplus" == methodToLower)
        _qmMethod = DFTBPLUS;

    else if ("pyscf" == methodToLower)
        _qmMethod = PYSCF;

    else if ("turbomole" == methodToLower)
        _qmMethod = TURBOMOLE;

    else if ("mace" == methodToLower)
        _qmMethod = MACE;

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
 * @brief sets the qmLoopTimeLimit in settings
 *
 * @param time
 */
void QMSettings::setQMLoopTimeLimit(const double time)
{
    _qmLoopTimeLimit = time;
}

/**
 * @brief sets the maceModel to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModel(const std::string_view &model)
{
    using enum MaceModelSize;
    const auto modelToLower = toLowerCopy(model);

    if ("large" == modelToLower)
        _maceModel = LARGE;

    else if ("medium" == modelToLower)
        _maceModel = MEDIUM;

    else if ("small" == modelToLower)
        _maceModel = SMALL;

    else
        throw UserInputException(
            std::format("Mace {} model not recognized", model)
        );
}

/**
 * @brief sets the maceModel to enum in settings
 *
 * @param model
 */
void QMSettings::setMaceModel(const MaceModelSize model) { _maceModel = model; }

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
 * @brief returns the qmLoopTimeLimit
 *
 * @return double
 */
double QMSettings::getQMLoopTimeLimit() { return _qmLoopTimeLimit; }

/**
 * @brief returns the maceModel
 *
 * @return MaceModelSize
 */
MaceModelSize QMSettings::getMaceModel() { return _maceModel; }