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

/**
 * @brief Set the Hessian file path.
 *
 * @param filename The path to the Hessian file.
 */
void HessianSettings::setHessianFile(const std::string &filename)
{
    _hessianFile.set(filename);
}

/**
 * @brief Set the Hessian info file path.
 *
 * @param filename The path to the Hessian info file.
 */
void HessianSettings::setHessianInfoFile(const std::string &filename)
{
    _hessianInfoFile.set(filename);
}

/**
 * @brief Set the displacement value for Hessian calculations.
 *
 * @param displacement The displacement value to be used.
 */
void HessianSettings::setDisplacement(double displacement)
{
    _displacement.set(displacement);
}

/**
 * @brief Set whether to optimize before Hessian calculations.
 *
 * @param optimize True to optimize before Hessian calculations, false
 * otherwise.
 */
void HessianSettings::setOptimizeBeforeHessian(bool optimize)
{
    _optimizeBeforeHessian.set(optimize);
}

/**
 * @brief Set the Hessian builder type.
 *
 * @param builder The Hessian builder type to be used.
 */
void HessianSettings::setBuilder(HessianBuilderType builder)
{
    _builder.set(builder);
}

/**
 * @brief Get the Hessian file path.
 *
 * @return The path to the Hessian file.
 */
std::string HessianSettings::getHessianFile() const
{
    return _hessianFile.get();
}

/**
 * @brief Get the Hessian info file path.
 *
 * @return The path to the Hessian info file.
 */
std::string HessianSettings::getHessianInfoFile() const
{
    return _hessianInfoFile.get();
}

/**
 * @brief Get the displacement value for Hessian calculations.
 *
 * @return The displacement value.
 */
double HessianSettings::getDisplacement() const { return _displacement.get(); }

/**
 * @brief Check whether to optimize before Hessian calculations.
 *
 * @return True if optimization before Hessian calculations is enabled,
 * false otherwise.
 */
bool HessianSettings::optimizeBeforeHessian() const
{
    return _optimizeBeforeHessian.get();
}

/**
 * @brief Get the Hessian builder type.
 *
 * @return The Hessian builder type.
 */
HessianBuilderType HessianSettings::getBuilder() const
{
    return _builder.get();
}
