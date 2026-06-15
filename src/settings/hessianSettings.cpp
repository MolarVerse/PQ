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

#include "stringUtilities.hpp"

using namespace settings;
using namespace utilities;

std::string settings::string(const HessianBuilderType builder)
{
    switch (builder)
    {
        using enum HessianBuilderType;

        case FINITE_DIFFERENCE_FORCES_CENTRAL: return "CENTRAL";
        case FINITE_DIFFERENCE_FORCES_FORWARD: return "FORWARD";
        case FINITE_DIFFERENCE_FORCES_FIVE_POINT: return "FIVE-POINT";
        case ANALYTIC: return "ANALYTIC";
        case NONE: return "NONE";

        default: return "NONE";
    }
}

void HessianSettings::setHessianFile(const std::string_view &filename)
{
    _hessianFile = filename;
}

void HessianSettings::setHessianInfoFile(const std::string_view &filename)
{
    _hessianInfoFile = filename;
}

void HessianSettings::setDisplacement(const double displacement)
{
    _displacement = displacement;
}

void HessianSettings::setOptimizeBeforeHessian(const bool optimize)
{
    _optimizeBeforeHessian = optimize;
}

void HessianSettings::setBuilder(const std::string_view &builder)
{
    using enum HessianBuilderType;

    const auto builderLower = toLowerAndReplaceDashesCopy(builder);

    if ("central" == builderLower)
        setBuilder(FINITE_DIFFERENCE_FORCES_CENTRAL);

    else if ("forward" == builderLower)
        setBuilder(FINITE_DIFFERENCE_FORCES_FORWARD);

    else if ("five_point" == builderLower)
        setBuilder(FINITE_DIFFERENCE_FORCES_FIVE_POINT);

    else if ("analytic" == builderLower)
        setBuilder(ANALYTIC);

    else
        setBuilder(NONE);
}

void HessianSettings::setBuilder(const HessianBuilderType builder)
{
    _builder = builder;
}

std::string HessianSettings::getHessianFile() { return _hessianFile; }

std::string HessianSettings::getHessianInfoFile() { return _hessianInfoFile; }

double HessianSettings::getDisplacement() { return _displacement; }

bool HessianSettings::optimizeBeforeHessian()
{
    return _optimizeBeforeHessian;
}

HessianBuilderType HessianSettings::getBuilder() { return _builder; }
