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

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "restartFileReader/restartFileSection.hpp"
#include "testRestartFileSection.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"

TEST_F(TestNoseHooverSection, testKeyword)
{
    EXPECT_EQ(_section->keyword(), "chi");
}

TEST_F(TestNoseHooverSection, testIsHeader)
{
    EXPECT_TRUE(_section->isHeader());
}

TEST_F(TestNoseHooverSection, testNumberOfArguments)
{
    size_t lineNumber = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        if (i != 4)
        {
            auto line = std::vector<std::string>(i);
            EXPECT_THROW_MSG(
                _section->process(line, *_engine, lineNumber),
                exc::RstFileException,
                "Error not enough arguments in line 0 for a chi entry of the "
                "nose hoover thermostat"
            );
        }
    }
}

/**
 * The restart file is 1-based (chi 1 ... chi N), the settings are 0-based.
 */
TEST_F(TestNoseHooverSection, testProcessConvertsToZeroBasedIndices)
{
    auto &nhSettings = _settings.noseHooverSettings;
    nhSettings.setNoseHooverChainLengthByUser(3);

    size_t lineNumber = 0;

    std::vector<std::string> line1 = {"chi", "1", "0.1", "0.4"};
    std::vector<std::string> line2 = {"chi", "2", "0.2", "0.5"};
    std::vector<std::string> line3 = {"chi", "3", "0.3", "0.6"};
    _section->process(line1, *_engine, lineNumber);
    _section->process(line2, *_engine, lineNumber);
    _section->process(line3, *_engine, lineNumber);

    const auto expectedChi =
        std::map<size_t, double>{{0, 0.1}, {1, 0.2}, {2, 0.3}};
    const auto expectedZeta =
        std::map<size_t, double>{{0, 0.4}, {1, 0.5}, {2, 0.6}};

    EXPECT_EQ(nhSettings.getChi(), expectedChi);
    EXPECT_EQ(nhSettings.getZeta(), expectedZeta);
}

TEST_F(TestNoseHooverSection, testProcessRejectsZeroIndex)
{
    _settings.noseHooverSettings.setNoseHooverChainLengthByUser(3);

    size_t                   lineNumber = 0;
    std::vector<std::string> line       = {"chi", "0", "0.1", "0.4"};

    EXPECT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Error in line 0 in restart file; chi index cannot be zero"
    );
}

TEST_F(TestNoseHooverSection, testProcessRejectsIndexLargerThanChainLength)
{
    _settings.noseHooverSettings.setNoseHooverChainLengthByUser(3);

    size_t                   lineNumber = 0;
    std::vector<std::string> line       = {"chi", "4", "0.1", "0.4"};

    EXPECT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Chi index 4 is larger than the number of nose hoover chains 3"
    );
}

TEST_F(TestNoseHooverSection, testProcessRejectsDuplicateIndex)
{
    _settings.noseHooverSettings.setNoseHooverChainLengthByUser(3);

    size_t                   lineNumber = 0;
    std::vector<std::string> line       = {"chi", "1", "0.1", "0.4"};

    _section->process(line, *_engine, lineNumber);

    EXPECT_THROW_MSG(
        _section->process(line, *_engine, lineNumber),
        exc::RstFileException,
        "Error in line 0 in restart file; chi or zeta entry already exists"
    );
}
