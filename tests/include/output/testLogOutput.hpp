/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

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