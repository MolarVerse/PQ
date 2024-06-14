/*****************************************************************************
<GPL_HEADER>

    PQ
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

#ifndef _TEST_SETUP_H_

#define _TEST_SETUP_H_

#include <gtest/gtest.h>

#include "engine.hpp"
#include "mmmdEngine.hpp"
#include "mmoptEngine.hpp"
#include "thermostatSettings.hpp"

class TestSetup : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        // NOTE: here the MMOPTEngine is used as dummy engine
        //       for testing the InputFileReader class
        //       The mdEngine is used only for special cases
        //       where optEngine is not supported
        _engine   = new engine::MMOptEngine();
        _mdEngine = new engine::MMMDEngine();
    }

    engine::Engine   *_engine;
    engine::MDEngine *_mdEngine;

    void TearDown() override
    {
        settings::ThermostatSettings::setEndTemperatureSet(false);
        settings::ThermostatSettings::setStartTemperatureSet(false);
        settings::ThermostatSettings::setTemperatureSet(false);
    }
};

#endif