#ifndef _TEST_MANOSTAT_HPP_

#define _TEST_MANOSTAT_HPP_

#include "manostat.hpp"
#include "physicalData.hpp"

#include <gtest/gtest.h>

/**
 * @class TestManostat
 *
 * @brief Fixture for manostat tests.
 *
 */
class TestManostat : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        _manostat = new manostat::Manostat();
        _data     = new physicalData::PhysicalData();

        _data->setVolume(2.0);
        _data->setVirial({1.0, 2.0, 3.0});
        _data->setKineticEnergyMolecularVector({1.0, 2.0, 3.0});
        _data->setKineticEnergyAtomicVector({1.0, 1.0, 1.0});

        _box = new simulationBox::SimulationBox();
    }

    void TearDown() override
    {
        delete _data;
        delete _box;
        delete _manostat;
    }

    physicalData::PhysicalData   *_data;
    simulationBox::SimulationBox *_box;
    manostat::Manostat           *_manostat;
};

#endif