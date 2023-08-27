#ifndef _TEST_INPUT_FILE_READER_H_

#define _TEST_INPUT_FILE_READER_H_

#include "engine.hpp"            // for Engine
#include "inputFileReader.hpp"   // for InputFileReader

#include <cstdio>          // for remove
#include <exception>       // for exception
#include <gtest/gtest.h>   // for Test
#include <iostream>        // for operator<<, basic_ostream, cerr, ostream
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
        _engine          = engine::Engine();
        _inputFileReader = new readInput::InputFileReader("input.in", _engine);
    }

    void TearDown() override
    {
        delete _inputFileReader;
        remove_file();
    }

    std::string _filename = "";

    engine::Engine              _engine;
    readInput::InputFileReader *_inputFileReader;

    void remove_file()
    {
        try
        {
            std::remove(_filename.c_str());
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
};

#endif