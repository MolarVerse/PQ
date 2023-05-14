#ifndef _TEST_BOX_H_

#define _TEST_BOX_H_

#include <gtest/gtest.h>

#include "box.hpp"

class TestBox : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        _box = new Box();
    }

    virtual void TearDown()
    {
        delete _box;
    }

    Box *_box;
};

#endif // _TEST_BOX_H_