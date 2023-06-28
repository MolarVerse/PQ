#ifndef _TEST_RST_FILE_READER_H_

#define _TEST_RST_FILE_READER_H_

#include "rstFileReader.hpp"

#include <gtest/gtest.h>

class TestRstFileReader : public ::testing::Test
{
  protected:
    void SetUp() override { _engine = engine::Engine(); }

    engine::Engine _engine;
};

#endif