#ifndef _TEST_POST_PROCESS_SETUP_H_

#define _TEST_POST_PROCESS_SETUP_H_

#include <gtest/gtest.h>

#include "postProcessSetup.hpp"

class TestPostProcessSetup : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _engine = Engine();
    }

    Engine _engine;
};

#endif