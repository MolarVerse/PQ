#ifndef _TEST_CELL_LIST_HPP_

#define _TEST_CELL_LIST_HPP_

#include "celllist.hpp"
#include "simulationBox.hpp"   // for SimulationBox

#include <gtest/gtest.h>

class TestCellList : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        _cellList = new simulationBox::CellList();
        _cellList->setNumberOfCells(2);
        _cellList->setNumberOfNeighbourCells(1);

        _simulationBox = new simulationBox::SimulationBox();
        _simulationBox->setBoxDimensions({10.0, 10.0, 10.0});
        _simulationBox->setCoulombRadiusCutOff(1.5);
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