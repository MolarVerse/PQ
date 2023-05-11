#ifndef _TEST_INPUT_FILE_READER_H_

#define _TEST_INPUT_FILE_READER_H_

#include <gtest/gtest.h>

#include "inputFileReader.hpp"

namespace Setup::InputFileReader
{
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
            _engine = Engine();
            _inputFileReader = new InputFileReader("input.in", _engine);
        }

        void TearDown() override
        {
            delete _inputFileReader;
        };

        Engine _engine;
        InputFileReader *_inputFileReader;
    };
}

#endif