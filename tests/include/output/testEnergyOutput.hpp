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

#ifndef TESTS__INCLUDE__OUTPUT__TEST_ENERGY_OUTPUT_HPP
#define TESTS__INCLUDE__OUTPUT__TEST_ENERGY_OUTPUT_HPP

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

#include "energyOutput.hpp"
#include "infoOutput.hpp"
#include "momentumOutput.hpp"
#include "physicalData.hpp"
#include "settings.hpp"

/**
 * @class TestEnergyOutput
 *
 * @brief test suite for energy output
 *
 */
class TestEnergyOutput : public ::testing::Test
{
   protected:
    std::unique_ptr<out::InfoOutput>            _infoOutput;
    std::unique_ptr<out::EnergyOutput>          _energyOutput;
    std::unique_ptr<out::MomentumOutput>        _momentumOutput;
    std::unique_ptr<physicalData::PhysicalData> _physicalData;
    std::unique_ptr<Settings>                   _settings;

    void SetUp() override
    {
        _infoOutput     = std::make_unique<out::InfoOutput>("default.info");
        _energyOutput   = std::make_unique<out::EnergyOutput>("default.en");
        _momentumOutput = std::make_unique<out::MomentumOutput>("default.mom");
        _physicalData   = std::make_unique<physicalData::PhysicalData>();
        _settings       = std::make_unique<Settings>();
    }

    void TearDown() override
    {
        static_cast<void>(std::filesystem::remove("default.info"));
        static_cast<void>(std::filesystem::remove("default.en"));
        static_cast<void>(std::filesystem::remove("default.mom"));
    }
};

#endif   // TESTS__INCLUDE__OUTPUT__TEST_ENERGY_OUTPUT_HPP
