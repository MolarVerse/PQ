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

#include <gtest/gtest.h>

#include <stdexcept>

#include "exceptions.hpp"
#include "hybridInputParser.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

using namespace customException;
using namespace input;

TEST_F(TestInputFileReader, parseInvalidQMCharges)
{
    auto parser = HybridInputParser(*_engine);

    EXPECT_THROW_MSG(
        parser.parseUseQMCharges({"qm_charges", "=", "invalid"}, 0),
        InputFileException,
        "Invalid qm_charges \"invalid\" in input file\n"
        "Possible values are: qm, mm"
    );
}

TEST_F(TestInputFileReader, parseNegativeHybridRadii)
{
    auto parser = HybridInputParser(*_engine);

    EXPECT_THROW_MSG(
        parser.parseCoreRadius({"core_radius", "=", "-1"}, 0),
        InputFileException,
        "Invalid core_radius -1 in input file - must be a positive number"
    );
    EXPECT_THROW_MSG(
        parser.parseLayerRadius({"layer_radius", "=", "-1"}, 0),
        InputFileException,
        "Invalid layer_radius -1 in input file - must be a positive number"
    );
    EXPECT_THROW_MSG(
        parser.parseSmoothingRadius({"smoothing_radius", "=", "-1"}, 0),
        InputFileException,
        "Invalid smoothing_radius -1 in input file - must be a positive number"
    );
}

TEST_F(TestInputFileReader, parseNonFiniteHybridRadii)
{
    auto parser = HybridInputParser(*_engine);

    EXPECT_THROW_MSG(
        parser.parseCoreRadius({"core_radius", "=", "nan"}, 0),
        std::invalid_argument,
        "Invalid floating-point value 'nan' encountered"
    );
    EXPECT_THROW_MSG(
        parser.parseLayerRadius({"layer_radius", "=", "inf"}, 0),
        std::invalid_argument,
        "Invalid floating-point value 'inf' encountered"
    );
    EXPECT_THROW_MSG(
        parser.parseSmoothingRadius({"smoothing_radius", "=", "-inf"}, 0),
        std::invalid_argument,
        "Invalid floating-point value '-inf' encountered"
    );
}
