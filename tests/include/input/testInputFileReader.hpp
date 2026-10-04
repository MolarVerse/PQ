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

#ifndef _TEST_INPUT_FILE_READER_H_

#define _TEST_INPUT_FILE_READER_H_

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "constraintSettings.hpp"
#include "inputConverter.hpp"
#include "inputFileParser.hpp"
#include "inputFileReader.hpp"
#include "mmmdEngine.hpp"
#include "optEngine.hpp"
#include "settings.hpp"

/**
 * @class TestInputFileReader
 *
 * @brief Test fixture for testing the InputFileReader class.
 *
 */
class TestInputFileReader : public ::testing::Test
{
   protected:
    std::string _fileName;

    std::unique_ptr<Settings>               _settings;
    std::unique_ptr<engine::Engine>         _engine;
    std::unique_ptr<input::InputFileReader> _inputFileReader;

    std::unique_ptr<engine::MDEngine> _mdEngine;

    void SetUp() override
    {
        _settings = std::make_unique<Settings>();
        // NOTE: here the MMOPTEngine is used as dummy engine
        //       for testing the InputFileReader class
        //       The mdEngine is used only for special cases
        //       where optEngine is not supported
        _engine          = std::make_unique<engine::OptEngine>();
        _mdEngine        = std::make_unique<engine::MMMDEngine>();
        _inputFileReader = std::make_unique<input::InputFileReader>("input.in");
    }

    void TearDown() override { _removeFile(); }

    static void _clearParser(input::InputFileParser &parser)
    {
        parser._clear();
        settings::ConstraintSettings::deactivateShake();
        settings::ConstraintSettings::deactivateMShake();
        settings::ConstraintSettings::deactivateDistanceConstraints();
    }

    void _removeFile() const
    {
        static_cast<void>(std::remove(_fileName.c_str()));
    }

    static auto _parseSelectionNoPython(
        input::Converter<input::SelectionTag> &converter,
        const std::string                     &key
    )
    {
        return converter._parseSelectionNoPython(key);
    }
};

#endif
