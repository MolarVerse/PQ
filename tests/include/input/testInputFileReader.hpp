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

#include <gtest/gtest.h>   // for Test

#include <cstdio>   // for remove
#include <string>   // for allocator, string

#include "inputFileReader.hpp"   // for InputFileReader
#include "mmmdEngine.hpp"        // for MDEngine
#include "mmoptEngine.hpp"       // for MDEngine

/**
 * @class TestInputFileReader
 *
 * @brief Test fixture for testing the InputFileReader class.
 *
 */
class TestInputFileReader : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        _inputFileReader = new input::InputFileReader("input.in", *_engine);

        // NOTE: here the MMOPTEngine is used as dummy engine
        //       for testing the InputFileReader class
        //       The mdEngine is used only for special cases
        //       where optEngine is not supported
        _engine   = new engine::MMOptEngine();
        _mdEngine = new engine::MMMDEngine();
        _inputFileReader_mdEngine =
            new input::InputFileReader("input.in", *_mdEngine);
    }

    void TearDown() override
    {
        delete _inputFileReader;
        removeFile();
    }

    std::string _fileName = "";

    engine::Engine         *_engine;
    input::InputFileReader *_inputFileReader;

    engine::MDEngine       *_mdEngine;
    input::InputFileReader *_inputFileReader_mdEngine;

    void removeFile() const { std::remove(_fileName.c_str()); }
};

#endif