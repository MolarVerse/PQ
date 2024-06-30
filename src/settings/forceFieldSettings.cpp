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

#include "forceFieldSettings.hpp"

using namespace settings;

/********************
 * standard getters *
 ********************/

/**
 * @brief Get if the force field is active
 *
 * @return ForceFieldType
 */
bool ForceFieldSettings::isActive() { return _active; }

/********************
 * standard setters *
 ********************/

/**
 * @brief set the force field active
 *
 */
void ForceFieldSettings::activate() { _active = true; }

/**
 * @brief set the force field inactive
 *
 */
void ForceFieldSettings::deactivate() { _active = false; }