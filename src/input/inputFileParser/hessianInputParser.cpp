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

#include "hessianInputParser.hpp"

#include <format>
#include <functional>

#include "exceptions.hpp"
#include "hessianSettings.hpp"
#include "stringUtilities.hpp"

using namespace input;
using namespace settings;
using namespace customException;
using namespace utilities;

HessianInputParser::HessianInputParser(pq::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("hessian_file"),
        std::bind_front(&HessianInputParser::parseHessianFile, this),
        false
    );
    addKeyword(
        std::string("hessian_info_file"),
        std::bind_front(&HessianInputParser::parseHessianInfoFile, this),
        false
    );
    addKeyword(
        std::string("hessian_displacement"),
        std::bind_front(&HessianInputParser::parseDisplacement, this),
        false
    );
    addKeyword(
        std::string("optimize_before_hessian"),
        std::bind_front(&HessianInputParser::parseOptimizeBeforeHessian, this),
        false
    );
    addKeyword(
        std::string("hessian_builder"),
        std::bind_front(&HessianInputParser::parseBuilder, this),
        false
    );
}

void HessianInputParser::parseHessianFile(
    const pq::strings &lineElements,
    const size_t       lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HessianSettings::setHessianFile(lineElements[2]);
}

void HessianInputParser::parseHessianInfoFile(
    const pq::strings &lineElements,
    const size_t       lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HessianSettings::setHessianInfoFile(lineElements[2]);
}

void HessianInputParser::parseDisplacement(
    const pq::strings &lineElements,
    const size_t       lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto displacement = stringToFiniteDouble(lineElements[2]);

    if (displacement <= 0.0)
        throw InputFileException(
            std::format(
                "Hessian displacement must be greater than 0 in input file "
                "at line {}",
                lineNumber
            )
        );

    HessianSettings::setDisplacement(displacement);
}

void HessianInputParser::parseOptimizeBeforeHessian(
    const pq::strings &lineElements,
    const size_t       lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HessianSettings::setOptimizeBeforeHessian(keywordToBool(lineElements));
}

void HessianInputParser::parseBuilder(
    const pq::strings &lineElements,
    const size_t       lineNumber
)
{
    using enum HessianBuilderType;

    checkCommand(lineElements, lineNumber);
    HessianSettings::setBuilder(lineElements[2]);

    if (HessianSettings::getBuilder() == NONE)
        throw InputFileException(
            std::format(
                "Invalid hessian_builder \"{}\" in input file at line {} - "
                "possible values are: central, forward, five-point, analytic",
                lineElements[2],
                lineNumber
            )
        );
}
