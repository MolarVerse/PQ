#ifndef _TEST_SETUP_H_

#define _TEST_SETUP_H_

#include "engine.hpp"

#include <gtest/gtest.h>

class TestSetup : public ::testing::Test
{
  protected:
    void SetUp() override { _engine = engine::Engine(); }

    engine::Engine _engine;
};

#endif