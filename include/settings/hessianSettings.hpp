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

#ifndef SETTINGS__HESSIAN_SETTINGS_HPP
#define SETTINGS__HESSIAN_SETTINGS_HPP

#include <string>

#include "enums/hessian.hpp"
#include "setting.hpp"
#include "unit_types.hpp"

/**
 * @brief Class representing the settings for Hessian calculations.
 *
 * This class encapsulates various settings related to Hessian calculations,
 * including file paths, displacement values, optimization flags, and the
 * Hessian builder type.
 */
class HessianSettings
{
   private:
    Setting<std::string, "Hessian File">                _hessianFile;
    Setting<std::string, "Hessian Info File">           _hessianInfoFile;
    Setting<Pos, "Displacement">                        _displacement;
    Setting<bool, "Optimize Before Hessian">            _optimizeBeforeHessian;
    Setting<HessianBuilderType, "Hessian Builder Type"> _builder;

   public:
    HessianSettings();

    SETTING_GETTER_AND_SETTER_BY_USER(_hessianFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_hessianInfoFile);
    SETTING_GETTER_AND_SETTER_BY_USER(_displacement);
    SETTING_GETTER_AND_SETTER_BY_USER(_optimizeBeforeHessian);
    SETTING_GETTER_AND_SETTER_BY_USER(_builder);
};

#endif   // SETTINGS__HESSIAN_SETTINGS_HPP
