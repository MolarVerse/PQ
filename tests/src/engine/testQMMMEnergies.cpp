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

#include <gtest/gtest.h>

#include "qmmmMDEngine.hpp"

class QMMMEnergiesTest : public ::testing::Test
{
   protected:
    static void _scaleAndAccumulate(engine::QMMMMDEngine &engine, double smF)
    {
        engine._scaleAndAccumulateEnergies(smF);
    }

    static void _moveEnergies(engine::QMMMMDEngine &engine)
    {
        engine._moveEnergiesToPhysicalData();
    }
};

TEST_F(QMMMEnergiesTest, exactSmoothingCountsIntraEnergyOnce)
{
    Settings             settings;
    engine::QMMMMDEngine engine{settings};
    auto                &data = *engine.getPhysicalDataPtr();

    data.resetEnergies();
    data.addCoulombEnergy(10.0);
    data.addIntraCoulombEnergy(1.0);
    data.addNonCoulombEnergy(20.0);
    data.addIntraNonCoulombEnergy(2.0);

    ASSERT_DOUBLE_EQ(data.getCoulombEnergy(), 11.0);
    ASSERT_DOUBLE_EQ(data.getNonCoulombEnergy(), 22.0);

    _scaleAndAccumulate(engine, 1.0);
    _moveEnergies(engine);

    EXPECT_DOUBLE_EQ(data.getCoulombEnergy(), 11.0);
    EXPECT_DOUBLE_EQ(data.getNonCoulombEnergy(), 22.0);
    EXPECT_DOUBLE_EQ(data.getIntraCoulombEnergy(), 1.0);
    EXPECT_DOUBLE_EQ(data.getIntraNonCoulombEnergy(), 2.0);
}
