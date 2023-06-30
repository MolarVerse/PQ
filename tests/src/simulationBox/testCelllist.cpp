#include "testCelllist.hpp"

TEST_F(TestCellList, determineCellSize)
{
    _cellList->determineCellSize(*_simulationBox);
    EXPECT_EQ(_cellList->getCellSize(), vector3d::Vec3D(5.0, 5.0, 5.0));
}

// TEST_F(TestCellList, determineCellBoundaries)
// {
//     _cellList->determineCellSize(*_simulationBox);
//     _cellList->determineCellBoundaries(*_simulationBox);

//     auto cells = _cellList->getCells();

//     const auto box   = _simulationBox->getBoxDimensions();
//     auto       index = static_cast<vector3d::Vec3D>(vector3d::Vec3Di(0, 0, 0));
//     EXPECT_EQ(cells[0].getLowerBoundary(), _cellList->getCellSize() * index - box / 2.0);
//     index = static_cast<vector3d::Vec3D>(vector3d::Vec3Di(0, 0, 1));
//     EXPECT_EQ(cells[0].getUpperBoundary(), _cellList->getCellSize() * index - box / 2.0);

//     index = static_cast<vector3d::Vec3D>(vector3d::Vec3Di(0, 0, 1));
//     EXPECT_EQ(cells[1].getLowerBoundary(), _cellList->getCellSize() * index - box / 2.0);
//     index = static_cast<vector3d::Vec3D>(vector3d::Vec3Di(0, 0, 2));
//     EXPECT_EQ(cells[1].getUpperBoundary(), _cellList->getCellSize() * index - box / 2.0);
// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}