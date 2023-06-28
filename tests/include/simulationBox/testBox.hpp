#ifndef _TEST_BOX_H_

#define _TEST_BOX_H_

#include "box.hpp"

#include <gtest/gtest.h>

class TestBox : public ::testing::Test
{
  protected:
    virtual void SetUp() { _box = new simulationBox::Box(); }

    virtual void TearDown() { delete _box; }

    simulationBox::Box *_box;
};

#endif   // _TEST_BOX_H_