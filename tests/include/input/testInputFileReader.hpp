/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "engine.hpp"            // for Engine
#include "inputFileReader.hpp"   // for InputFileReader

#include <cstdio>          // for remove
#include <gtest/gtest.h>   // for Test
#include <string>          // for allocator, string

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
        _engine          = new engine::Engine();
        _inputFileReader = new input::InputFileReader("input.in", *_engine);
    }

    void TearDown() override
    {
        delete _inputFileReader;
        removeFile();
    }

    std::string _fileName = "";

    engine::Engine             *_engine;
    input::InputFileReader *_inputFileReader;

    void removeFile() const { std::remove(_fileName.c_str()); }
};

#endif