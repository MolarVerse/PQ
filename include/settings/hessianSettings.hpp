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

#ifndef _HESSIAN_SETTINGS_HPP_

#define _HESSIAN_SETTINGS_HPP_

#include <string>

#include "enums/hessian.hpp"
#include "setting.hpp"

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
    Setting<std::string>        _hessianFile;
    Setting<std::string>        _hessianInfoFile;
    Setting<double>             _displacement;
    Setting<bool>               _optimizeBeforeHessian;
    Setting<HessianBuilderType> _builder;

   public:
    HessianSettings();

    void setHessianFile(const std::string &filename);
    void setHessianInfoFile(const std::string &filename);
    void setDisplacement(double displacement);
    void setOptimizeBeforeHessian(bool optimize);
    void setBuilder(HessianBuilderType builder);

    [[nodiscard]] std::string        getHessianFile() const;
    [[nodiscard]] std::string        getHessianInfoFile() const;
    [[nodiscard]] double             getDisplacement() const;
    [[nodiscard]] bool               optimizeBeforeHessian() const;
    [[nodiscard]] HessianBuilderType getBuilder() const;
};

#endif   // _HESSIAN_SETTINGS_HPP_
