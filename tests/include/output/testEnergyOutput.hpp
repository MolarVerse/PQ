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

#ifndef _TEST_ENERGY_OUTPUT_HPP_

#define _TEST_ENERGY_OUTPUT_HPP_

#include "energyOutput.hpp"     // for EnergyOutput
#include "infoOutput.hpp"       // for InfoOutput
#include "momentumOutput.hpp"   // for MomentumOutput
#include "physicalData.hpp"     // for PhysicalDat

#include <cstdio>          // for remove
#include <gtest/gtest.h>   // for Test
#include <memory>          // for allocator

/**
 * @class TestEnergyOutput
 *
 * @brief test suite for energy output
 *
 */
class TestEnergyOutput : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _infoOutput     = new output::InfoOutput("default.info");
        _energyOutput   = new output::EnergyOutput("default.en");
        _momentumOutput = new output::MomentumOutput("default.mom");
        _physicalData   = new physicalData::PhysicalData();
    }

    void TearDown() override
    {
        delete _infoOutput;
        delete _energyOutput;
        delete _momentumOutput;
        delete _physicalData;
        ::remove("default.info");
        ::remove("default.en");
        ::remove("default.mom");
    }

    output::InfoOutput         *_infoOutput;
    output::EnergyOutput       *_energyOutput;
    output::MomentumOutput     *_momentumOutput;
    physicalData::PhysicalData *_physicalData;
};

#endif   // _TEST_ENERGY_OUTPUT_HPP_