#ifndef _TEST_CELLLIST_HPP_

#define _TEST_CELLLIST_HPP_

#include "celllist.hpp"

#include <gtest/gtest.h>

class TestCellList : public ::testing::Test
{
  protected:
    virtual void SetUp()
    {
        _cellList = new simulationBox::CellList();
        _cellList->setNumberOfCells(2);

        _simulationBox = new simulationBox::SimulationBox();
        _simulationBox->setBoxDimensions({10.0, 10.0, 10.0});

        simulationBox::Cell *cell1 = new simulationBox::Cell();
        simulationBox::Cell *cell2 = new simulationBox::Cell();
        simulationBox::Cell *cell3 = new simulationBox::Cell();
        simulationBox::Cell *cell4 = new simulationBox::Cell();
        simulationBox::Cell *cell5 = new simulationBox::Cell();
        simulationBox::Cell *cell6 = new simulationBox::Cell();
        simulationBox::Cell *cell7 = new simulationBox::Cell();
        simulationBox::Cell *cell8 = new simulationBox::Cell();

        _cellList->addCellPointers(*cell1);
        _cellList->addCellPointers(*cell2);
        _cellList->addCellPointers(*cell3);
        _cellList->addCellPointers(*cell4);
        _cellList->addCellPointers(*cell5);
        _cellList->addCellPointers(*cell6);
        _cellList->addCellPointers(*cell7);
        _cellList->addCellPointers(*cell8);
    }

    virtual void TearDown()
    {
        delete _cellList;
        delete _simulationBox;
    }

    simulationBox::CellList      *_cellList;
    simulationBox::SimulationBox *_simulationBox;
};

#endif   // _TEST_CELLLIST_HPP_