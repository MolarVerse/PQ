#include "testCelllist.hpp"

using namespace std;

TEST_F(TestCellList, determineCellSize)
{
    _cellList->determineCellSize(*_simulationBox);
    EXPECT_EQ(_cellList->getCellSize(), vector3d::Vec3D(5.0, 5.0, 5.0));
}

TEST_F(TestCellList, determineCellBoundaries)
{
    _cellList->determineCellSize(*_simulationBox);
    _cellList->resizeCells(prod(_cellList->getNumberOfCells()));
    _cellList->determineCellBoundaries(*_simulationBox);

    auto cells = _cellList->getCells();

    const auto box   = _simulationBox->getBoxDimensions();
    auto       index = static_cast<vector3d::Vec3D>(cells[0].getCellIndex());
    EXPECT_EQ(cells[0].getLowerBoundary(), _cellList->getCellSize() * index - box / 2.0);
    EXPECT_EQ(cells[0].getUpperBoundary(), _cellList->getCellSize() * (index + 1.0) - box / 2.0);

    index = static_cast<vector3d::Vec3D>(cells[1].getCellIndex());
    EXPECT_EQ(cells[1].getLowerBoundary(), _cellList->getCellSize() * index - box / 2.0);
    EXPECT_EQ(cells[1].getUpperBoundary(), _cellList->getCellSize() * (index + 1.0) - box / 2.0);
}

TEST_F(TestCellList, getCellIndex)
{

    const auto cellIndices = vector3d::Vec3Dul(1, 2, 3);
    _cellList->getCellIndex(cellIndices);

    EXPECT_EQ(_cellList->getCellIndex(cellIndices), 1 * 2 * 2 + 2 * 2 + 3);
}

TEST_F(TestCellList, getCellIndexOfMolecule)
{
    const auto position1 = vector3d::Vec3D(1.0, 2.0, 3.0);
    const auto position2 = vector3d::Vec3D(6.0, 7.0, 8.0);

    _cellList->determineCellSize(*_simulationBox);

    EXPECT_EQ(_cellList->getCellIndexOfMolecule(*_simulationBox, position1), vector3d::Vec3Dul(1, 1, 1));
    EXPECT_EQ(_cellList->getCellIndexOfMolecule(*_simulationBox, position2), vector3d::Vec3Dul(0, 0, 0));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}