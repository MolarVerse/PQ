#ifndef _TEST_STDOUTOUTPUT_HPP_

#define _TEST_STDOUTOUTPUT_HPP_

#include "stdoutOutput.hpp"

#include <gtest/gtest.h>

class TestStdoutOutput : public ::testing::Test
{
  protected:
    void SetUp() override { _stdoutOutput = new output::StdoutOutput("stdout"); }

    void TearDown() override { delete _stdoutOutput; }

    output::StdoutOutput *_stdoutOutput;
};

#endif   // _TEST_STDOUTOUTPUT_HPP_