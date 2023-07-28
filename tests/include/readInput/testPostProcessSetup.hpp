#ifndef _TEST_POST_PROCESS_SETUP_H_

#define _TEST_POST_PROCESS_SETUP_H_

#include "postProcessSetup.hpp"

#include <gtest/gtest.h>

class TestPostProcessSetup : public ::testing::Test
{
  protected:
    void SetUp() override { _engine = engine::Engine(); }

    engine::Engine _engine;
};

#endif