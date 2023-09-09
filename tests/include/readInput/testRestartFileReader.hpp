#ifndef _TEST_RST_FILE_READER_H_

#define _TEST_RST_FILE_READER_H_

#include "engine.hpp"   // for Engine

#include <gtest/gtest.h>

/**
 * @brief Fixture class for testing the RestartFileReader class
 *
 */
class TestRstFileReader : public ::testing::Test
{
  protected:
    void SetUp() override { _engine = new engine::Engine(); }

    engine::Engine *_engine;
};

#endif