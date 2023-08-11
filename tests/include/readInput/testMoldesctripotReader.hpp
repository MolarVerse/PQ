#ifndef _TEST_MOLDESCRIPTOR_READER_H_

#define _TEST_MOLDESCRIPTOR_READER_H_

#include "moldescriptorReader.hpp"

#include <gtest/gtest.h>

/**
 * @class TestMoldescriptorReader
 *
 * @brief Fixture class for testing the MoldescriptorReader class
 *
 */
class TestMoldescriptorReader : public ::testing::Test
{
  protected:
    void SetUp() override { _engine = engine::Engine(); }

    engine::Engine _engine;
};

#endif