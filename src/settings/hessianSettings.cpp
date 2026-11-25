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

#include "hessianSettings.hpp"

#include "defaults.hpp"

/**
 * @brief Construct a new HessianSettings object with default values.
 *
 * Initializes the Hessian settings with default file paths, displacement,
 * optimization flag, and Hessian builder type.
 */
HessianSettings::HessianSettings()
    : _hessianFile(DefaultFiles::hessianFile),
      _hessianInfoFile(DefaultFiles::hessianInfoFile),
      _displacement(defaults::HESSIAN_DISPLACEMENT_DEFAULT),
      _optimizeBeforeHessian(defaults::HESSIAN_OPTIMIZE_DEFAULT),
      _builder(HessianBuilderType::CENTRAL)
{
}
