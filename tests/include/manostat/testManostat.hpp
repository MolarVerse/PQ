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

#ifndef _TEST_MANOSTAT_HPP_

#define _TEST_MANOSTAT_HPP_

#include <gtest/gtest.h>

#include "manostat.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

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
        _data->setVirial(diagonalMatrix(linearAlgebra::Vec3D(1.0, 2.0, 3.0)));
        _data->setKineticEnergyMolecularVector(
            diagonalMatrix(linearAlgebra::Vec3D(1.0, 2.0, 3.0))
        );
        _data->setKineticEnergyAtomicVector(
            diagonalMatrix(linearAlgebra::Vec3D(1.0, 1.0, 1.0))
        );

        _box = new simulationBox::SimulationBox();
        _box->setVolume(2.0);
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