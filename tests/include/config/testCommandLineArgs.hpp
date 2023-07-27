#ifndef _TEST_COMMAND_LINE_ARGS_HPP_

#define _TEST_COMMAND_LINE_ARGS_HPP_

#include "commandLineArgs.hpp"

#include <gtest/gtest.h>

/**
 * @class TestCommandLineArgs
 *
 * @brief contains tests for CommandLineArgs class
 *
 */
class TestCommandLineArgs : public ::testing::Test
{
  protected:
    CommandLineArgs *_commandLineArgs;

    void SetUp() override {}

    void TearDown() override { delete _commandLineArgs; }
};

#endif   // _TEST_COMMAND_LINE_ARGS_HPP_