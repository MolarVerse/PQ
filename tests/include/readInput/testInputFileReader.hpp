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
        _inputFileReader = new readInput::InputFileReader("input.in", *_engine);
    }

    void TearDown() override
    {
        delete _inputFileReader;
        removeFile();
    }

    std::string _fileName = "";

    engine::Engine             *_engine;
    readInput::InputFileReader *_inputFileReader;

    void removeFile() const { std::remove(_fileName.c_str()); }
};

#endif