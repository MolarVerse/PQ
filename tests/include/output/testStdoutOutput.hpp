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

#ifndef _TEST_STDOUTOUTPUT_HPP_

#define _TEST_STDOUTOUTPUT_HPP_

#include "stdoutOutput.hpp"   // for StdoutOutput

#include <gtest/gtest.h>   // for Test
#include <memory>          // for allocator

/**
 * @class TestStdoutOutput
 *
 * @brief test suite for stdout output
 *
 */
class TestStdoutOutput : public ::testing::Test
{
  protected:
    void SetUp() override { _stdoutOutput = new output::StdoutOutput("stdout"); }

    void TearDown() override { delete _stdoutOutput; }

    output::StdoutOutput *_stdoutOutput;
};

#endif   // _TEST_STDOUTOUTPUT_HPP_