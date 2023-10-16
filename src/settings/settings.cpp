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

#include "settings.hpp"

#include "stringUtilities.hpp"   // for toLowerCopy

#include <string>   // for operator==, string

using settings::Settings;

/**
 * @brief sets the jobtype to enum in settings
 *
 * @param jobtype
 */
void Settings::setJobtype(const std::string_view jobtype)
{
    const auto jobtypeToLower = utilities::toLowerCopy(jobtype);

    if (jobtypeToLower == "mmmd")
        _jobtype = settings::JobType::MM_MD;
    else if (jobtypeToLower == "qmmd")
        _jobtype = settings::JobType::QM_MD;
    else if (jobtypeToLower == "ring_polymer_qmmd")
        _jobtype = settings::JobType::RING_POLYMER_QM_MD;
    else
        _jobtype = settings::JobType::NONE;
}

/**
 * @brief Returns true if the jobtype does no use any MM type simulations
 */
bool Settings::isQMOnly()
{
    if (_jobtype == settings::JobType::QM_MD)
        return true;
    else if (_jobtype == settings::JobType::RING_POLYMER_QM_MD)
        return true;
    else
        return false;
}