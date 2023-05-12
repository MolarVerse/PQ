#ifndef _TEST_MOLDESCRIPTOR_READER_H_

#define _TEST_MOLDESCRIPTOR_READER_H_

#include <gtest/gtest.h>

#include "moldescriptorReader.hpp"

class TestMoldescriptorReader : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _engine = Engine();
    }

    Engine _engine;
};

#endif