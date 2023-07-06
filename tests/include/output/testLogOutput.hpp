#ifndef _TEST_LOGOUTPUT_HPP_

#define _TEST_LOGOUTPUT_HPP_

#include "logOutput.hpp"

#include <gtest/gtest.h>

class TestLogOutput : public ::testing::Test
{
  protected:
    void SetUp() override { _logOutput = new output::LogOutput("default.out"); }

    void TearDown() override
    {
        delete _logOutput;
        remove("default.out");
    }

    output::LogOutput *_logOutput;
};

#endif   // _TEST_LOGOUTPUT_HPP_