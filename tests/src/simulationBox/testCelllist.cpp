#include "testCelllist.hpp"

#include "atom.hpp"               // for Atom
#include "cell.hpp"               // for Cell
#include "exceptions.hpp"         // for CellListException
#include "molecule.hpp"           // for Molecule
#include "simulationBox.hpp"      // for SimulationBox
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG
#include "vector3d.hpp"           // for Vec3Dul, Vec3D, Vector3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <memory>          // for make_shared, __shared_ptr_access
#include <string>          // for allocator, basic_string
#include <vector>          // for vector

TEST_F(TestCellList, determineCellSize)
{
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    EXPECT_EQ(_cellList->getCellSize(), linearAlgebra::Vec3D(5.0, 5.0, 5.0));
}

TEST_F(TestCellList, determineCellBoundaries)
{
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    _cellList->resizeCells();
    _cellList->determineCellBoundaries(_simulationBox->getBoxDimensions());

    auto cells = _cellList->getCells();

    const auto box   = _simulationBox->getBoxDimensions();
    auto       index = static_cast<linearAlgebra::Vec3D>(cells[0].getCellIndex());
    EXPECT_EQ(cells[0].getLowerBoundary(), _cellList->getCellSize() * index - box / 2.0);
    EXPECT_EQ(cells[0].getUpperBoundary(), _cellList->getCellSize() * (index + 1.0) - box / 2.0);

    index = static_cast<linearAlgebra::Vec3D>(cells[1].getCellIndex());
    EXPECT_EQ(cells[1].getLowerBoundary(), _cellList->getCellSize() * index - box / 2.0);
    EXPECT_EQ(cells[1].getUpperBoundary(), _cellList->getCellSize() * (index + 1.0) - box / 2.0);
}

TEST_F(TestCellList, getCellIndex)
{
    const auto                  cellIndices = linearAlgebra::Vec3Dul(1, 2, 3);
    [[maybe_unused]] const auto dummy       = _cellList->getCellIndex(cellIndices);

    EXPECT_EQ(_cellList->getCellIndex(cellIndices), 1 * 2 * 2 + 2 * 2 + 3);
}

TEST_F(TestCellList, getCellIndexOfAtom)
{
    const auto position1 = linearAlgebra::Vec3D(1.0, 2.0, 3.0);
    const auto position2 = linearAlgebra::Vec3D(6.0, 7.0, 8.0);

    _cellList->determineCellSize(_simulationBox->getBoxDimensions());

    EXPECT_EQ(_cellList->getCellIndexOfAtom(_simulationBox->getBoxDimensions(), position1), linearAlgebra::Vec3Dul(1, 1, 1));
    EXPECT_EQ(_cellList->getCellIndexOfAtom(_simulationBox->getBoxDimensions(), position2), linearAlgebra::Vec3Dul(0, 0, 0));
}

TEST_F(TestCellList, addNeighbouringCellPointers)
{
    auto cell = simulationBox::Cell();
    cell.setCellIndex(linearAlgebra::Vec3Dul(0, 0, 0));

    _cellList->setNumberOfCells(7);
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    _cellList->resizeCells();
    _cellList->determineCellBoundaries(_simulationBox->getBoxDimensions());
    _cellList->addNeighbouringCellPointers(cell);

    const auto neighbourCells = cell.getNeighbourCells();

    EXPECT_EQ(neighbourCells.size(), 13);
    EXPECT_EQ(neighbourCells[0]->getCellIndex(), linearAlgebra::Vec3Dul(6, 6, 6));
    EXPECT_EQ(neighbourCells[1]->getCellIndex(), linearAlgebra::Vec3Dul(6, 6, 0));
    EXPECT_EQ(neighbourCells[2]->getCellIndex(), linearAlgebra::Vec3Dul(6, 6, 1));
    EXPECT_EQ(neighbourCells[3]->getCellIndex(), linearAlgebra::Vec3Dul(6, 0, 6));
    EXPECT_EQ(neighbourCells[4]->getCellIndex(), linearAlgebra::Vec3Dul(6, 0, 0));
    EXPECT_EQ(neighbourCells[5]->getCellIndex(), linearAlgebra::Vec3Dul(6, 0, 1));
    EXPECT_EQ(neighbourCells[6]->getCellIndex(), linearAlgebra::Vec3Dul(6, 1, 6));
    EXPECT_EQ(neighbourCells[7]->getCellIndex(), linearAlgebra::Vec3Dul(6, 1, 0));
    EXPECT_EQ(neighbourCells[8]->getCellIndex(), linearAlgebra::Vec3Dul(6, 1, 1));
    EXPECT_EQ(neighbourCells[9]->getCellIndex(), linearAlgebra::Vec3Dul(0, 6, 6));
    EXPECT_EQ(neighbourCells[10]->getCellIndex(), linearAlgebra::Vec3Dul(0, 6, 0));
    EXPECT_EQ(neighbourCells[11]->getCellIndex(), linearAlgebra::Vec3Dul(0, 6, 1));
    EXPECT_EQ(neighbourCells[12]->getCellIndex(), linearAlgebra::Vec3Dul(0, 0, 6));
}

TEST_F(TestCellList, addNeighbouringCells)
{
    _cellList->setNumberOfCells(7);
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    _cellList->resizeCells();
    _cellList->determineCellBoundaries(_simulationBox->getBoxDimensions());
    _cellList->addNeighbouringCells(_simulationBox->getCoulombRadiusCutOff());

    for (const auto &cell : _cellList->getCells())
    {
        const auto neighbourCells = cell.getNeighbourCells();
        EXPECT_EQ(neighbourCells.size(), 62);
    }

    EXPECT_EQ(_cellList->getNumberOfNeighbourCells(), linearAlgebra::Vec3Dul(2, 2, 2));
}

/**
 * @brief testing checkCoulombCutoff method
 *
 */
TEST_F(TestCellList, checkCoulombCutoff)
{
    _simulationBox->setBoxDimensions(linearAlgebra::Vec3D(50.0, 50.0, 50.0));
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    EXPECT_NO_THROW(_cellList->checkCoulombCutoff({200.0}));

    EXPECT_THROW_MSG(_cellList->checkCoulombCutoff({0.1}),
                     customException::CellListException,
                     "Coulomb cutoff is smaller than half of the largest cell size.");
}

/**
 * @brief testing updateCellList and setup method
 *
 * TODO: think of a clever way to break this test into smaller tests
 *
 */
TEST_F(TestCellList, updateCellList)
{
    _simulationBox->setCoulombRadiusCutOff(22.0);
    EXPECT_NO_THROW(_cellList->updateCellList(*_simulationBox));
    _cellList->activate();

    auto molecule = simulationBox::Molecule();
    molecule.setNumberOfAtoms(2);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();

    atom1->setPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
    atom2->setPosition(linearAlgebra::Vec3D(6.0, 7.0, 8.0));

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    _simulationBox->addMolecule(molecule);

    _cellList->setup(*_simulationBox);
    auto cellSizeOld = _cellList->getCellSize();

    _simulationBox->setBoxDimensions(linearAlgebra::Vec3D(50.0, 50.0, 50.0));
    _simulationBox->setBoxSizeHasChanged(true);

    _cellList->updateCellList(*_simulationBox);

    EXPECT_NE(_cellList->getCellSize(), cellSizeOld);

    _simulationBox->setBoxSizeHasChanged(false);
    cellSizeOld = _cellList->getCellSize();
    _cellList->updateCellList(*_simulationBox);

    EXPECT_EQ(_cellList->getCellSize(), cellSizeOld);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}