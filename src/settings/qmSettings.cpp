/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "stringUtilities.hpp"   // for toLowerCopy

using settings::QMMethod;
using settings::QMSettings;

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
    case QMMethod::DFTBPLUS: return "DFTBPLUS";
    case QMMethod::PYSCF: return "PYSCF";
    default: return "none";
    }
}

/**
 * @brief sets the qmMethod to enum in settings
 *
 * @param method
 */
void QMSettings::setQMMethod(const std::string_view &method)
{
    const auto methodToLower = utilities::toLowerCopy(method);

    if ("dftbplus" == method)
        _qmMethod = QMMethod::DFTBPLUS;
    else if ("pyscf" == method)
        _qmMethod = QMMethod::PYSCF;
    else
        _qmMethod = QMMethod::NONE;
}