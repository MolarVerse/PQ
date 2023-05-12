#ifndef _TEST_RST_FILE_READER_H_

#define _TEST_RST_FILE_READER_H_

#include <gtest/gtest.h>

#include "rstFileReader.hpp"

namespace Setup::RstFileReader
{
    class TestRstFileReader : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            _engine = Engine();
        }

        Engine _engine;
    };
}

#endif