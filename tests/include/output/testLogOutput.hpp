#ifndef _TEST_LOGOUTPUT_HPP_

#define _TEST_LOGOUTPUT_HPP_

#include "logOutput.hpp"   // for LogOutput

#include <gtest/gtest.h>   // for Test
#include <memory>          // for allocator
#include <stdio.h>         // for remove

/**
 * @class TestLogOutput
 *
 * @brief test suite for log output
 *
 */
class TestLogOutput : public ::testing::Test
{
  protected:
    void SetUp() override { _logOutput = new output::LogOutput("default.out"); }

    void TearDown() override
    {
        delete _logOutput;
        ::remove("default.out");
    }

    output::LogOutput *_logOutput;
};

#endif   // _TEST_LOGOUTPUT_HPP_