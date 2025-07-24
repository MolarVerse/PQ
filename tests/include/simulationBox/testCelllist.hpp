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

#ifndef _TEST_CELL_LIST_HPP_

#define _TEST_CELL_LIST_HPP_

#include <gtest/gtest.h>

#include "celllist.hpp"            // for CellList
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox

class TestCellList : public ::testing::Test
{
   protected:
    virtual void SetUp()
    {
        _cellList = new simulationBox::CellList();
        _cellList->setNumberOfCells(2);
        _cellList->setNumberOfNeighbourCells(1);
        _cellList->resizeCells();

        _simulationBox = new simulationBox::SimulationBox();
        _simulationBox->setBoxDimensions({10.0, 10.0, 10.0});
        settings::PotentialSettings::setCoulombRadiusCutOff(1.5);
    }

    virtual void TearDown()
    {
        delete _cellList;
        delete _simulationBox;
    }

    simulationBox::CellList      *_cellList;
    simulationBox::SimulationBox *_simulationBox;
};

#endif   // _TEST_CELL_LIST_HPP_