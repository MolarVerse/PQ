#ifndef _TEST_ENERGY_OUTPUT_HPP_

#define _TEST_ENERGY_OUTPUT_HPP_

#include "energyOutput.hpp"   // for EnergyOutput
#include "infoOutput.hpp"     // for InfoOutput
#include "physicalData.hpp"   // for PhysicalDat

#include <gtest/gtest.h>   // for Test
#include <memory>          // for allocator
#include <stdio.h>         // for remove

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
        _infoOutput   = new output::InfoOutput("default.info");
        _energyOutput = new output::EnergyOutput("default.en");
        _physicalData = new physicalData::PhysicalData();

        _physicalData->setTemperature(1.0);
        _physicalData->setPressure(2.0);
        _physicalData->setKineticEnergy(3.0);
        _physicalData->setCoulombEnergy(4.0);
        _physicalData->setNonCoulombEnergy(5.0);
        _physicalData->setMomentum(6.0);
    }

    void TearDown() override
    {
        delete _infoOutput;
        delete _energyOutput;
        delete _physicalData;
        ::remove("default.info");
        ::remove("default.en");
    }

    output::InfoOutput         *_infoOutput;
    output::EnergyOutput       *_energyOutput;
    physicalData::PhysicalData *_physicalData;
};

#endif   // _TEST_ENERGY_OUTPUT_HPP_