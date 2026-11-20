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

#ifndef TESTS__INCLUDE__PRIVATE_HELPER__POTENTIAL_SETTINGS_UTILS_HPP
#define TESTS__INCLUDE__PRIVATE_HELPER__POTENTIAL_SETTINGS_UTILS_HPP

#include "potentialSettings.hpp"

struct PotentialSettingsUtils
{
    static void setEnergyCutoff(
        PotentialSettings& settings,
        double             energyCutoff
    );

    static void setForceCutoff(PotentialSettings& settings, double forceCutoff);
};

inline void PotentialSettingsUtils::setEnergyCutoff(
    PotentialSettings& settings,
    double             energyCutoff
)
{
    settings._coulombConfig.energyCutoff = energyCutoff;
}

inline void PotentialSettingsUtils::setForceCutoff(
    PotentialSettings& settings,
    double             forceCutoff
)
{
    settings._coulombConfig.forceCutoff = forceCutoff;
}

#endif   // TESTS__INCLUDE__PRIVATE_HELPER__POTENTIAL_SETTINGS_UTILS_HPP
