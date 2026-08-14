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

#include "testCelllist.hpp"

#include <limits>   // for numeric_limits
#include <memory>   // for make_shared, __shared_ptr_access
#include <vector>   // for vector

#include "atom.hpp"                // for Atom
#include "cell.hpp"                // for Cell
#include "exceptions.hpp"          // for CellListException
#include "gtest/gtest.h"           // for Message, TestPartResult
#include "molecule.hpp"            // for Molecule
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "throwWithMessage.hpp"    // for EXPECT_THROW_MSG
#include "vector3d.hpp"   // IWYU pragma: keep - for Vec3Dul, Vec3D, Vector3D

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

    const auto &cells = _cellList->getCells();

    const auto box = _simulationBox->getBoxDimensions();
    auto index     = static_cast<linearAlgebra::Vec3D>(cells[0].getCellIndex());
    EXPECT_EQ(
        cells[0].getLowerBoundary(),
        _cellList->getCellSize() * index - box / 2.0
    );
    EXPECT_EQ(
        cells[0].getUpperBoundary(),
        _cellList->getCellSize() * (index + 1.0) - box / 2.0
    );

    index = static_cast<linearAlgebra::Vec3D>(cells[1].getCellIndex());
    EXPECT_EQ(
        cells[1].getLowerBoundary(),
        _cellList->getCellSize() * index - box / 2.0
    );
    EXPECT_EQ(
        cells[1].getUpperBoundary(),
        _cellList->getCellSize() * (index + 1.0) - box / 2.0
    );
}

TEST_F(TestCellList, getCellIndex)
{
    const auto                  cellIndices = linearAlgebra::Vec3Dul(1, 2, 3);
    [[maybe_unused]] const auto dummy = _cellList->getCellIndex(cellIndices);

    EXPECT_EQ(_cellList->getCellIndex(cellIndices), 1 * 2 * 2 + 2 * 2 + 3);
}

TEST_F(TestCellList, getCellIndexOfAtom)
{
    const auto position1 = linearAlgebra::Vec3D(1.0, 2.0, 3.0);
    const auto position2 = linearAlgebra::Vec3D(6.0, 7.0, 8.0);

    _cellList->determineCellSize(_simulationBox->getBoxDimensions());

    EXPECT_EQ(
        _cellList
            ->getCellIndexOfAtom(_simulationBox->getBoxDimensions(), position1),
        linearAlgebra::Vec3Dul(1, 1, 1)
    );
    EXPECT_EQ(
        _cellList
            ->getCellIndexOfAtom(_simulationBox->getBoxDimensions(), position2),
        linearAlgebra::Vec3Dul(0, 0, 0)
    );
}

TEST_F(TestCellList, getCellIndexOfAtom_wrapsPeriodicBoundaryCoordinates)
{
    _simulationBox->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));
    _cellList->setNumberOfCells(2);
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());

    EXPECT_EQ(
        _cellList->getCellIndexOfAtom(
            _simulationBox->getBoxDimensions(),
            linearAlgebra::Vec3D(-5.0, -5.0, -5.0)
        ),
        linearAlgebra::Vec3Dul(0, 0, 0)
    );
    EXPECT_EQ(
        _cellList->getCellIndexOfAtom(
            _simulationBox->getBoxDimensions(),
            linearAlgebra::Vec3D(0.0, 0.0, 0.0)
        ),
        linearAlgebra::Vec3Dul(1, 1, 1)
    );
    EXPECT_EQ(
        _cellList->getCellIndexOfAtom(
            _simulationBox->getBoxDimensions(),
            linearAlgebra::Vec3D(5.0, 5.0, 5.0)
        ),
        linearAlgebra::Vec3Dul(0, 0, 0)
    );
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

    const auto &neighbourCells = cell.getNeighbourCells();

    EXPECT_EQ(neighbourCells.size(), 13);
    EXPECT_EQ(
        neighbourCells[0]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 6, 6)
    );
    EXPECT_EQ(
        neighbourCells[1]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 6, 0)
    );
    EXPECT_EQ(
        neighbourCells[2]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 6, 1)
    );
    EXPECT_EQ(
        neighbourCells[3]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 0, 6)
    );
    EXPECT_EQ(
        neighbourCells[4]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 0, 0)
    );
    EXPECT_EQ(
        neighbourCells[5]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 0, 1)
    );
    EXPECT_EQ(
        neighbourCells[6]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 1, 6)
    );
    EXPECT_EQ(
        neighbourCells[7]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 1, 0)
    );
    EXPECT_EQ(
        neighbourCells[8]->getCellIndex(),
        linearAlgebra::Vec3Dul(6, 1, 1)
    );
    EXPECT_EQ(
        neighbourCells[9]->getCellIndex(),
        linearAlgebra::Vec3Dul(0, 6, 6)
    );
    EXPECT_EQ(
        neighbourCells[10]->getCellIndex(),
        linearAlgebra::Vec3Dul(0, 6, 0)
    );
    EXPECT_EQ(
        neighbourCells[11]->getCellIndex(),
        linearAlgebra::Vec3Dul(0, 6, 1)
    );
    EXPECT_EQ(
        neighbourCells[12]->getCellIndex(),
        linearAlgebra::Vec3Dul(0, 0, 6)
    );
}

TEST_F(TestCellList, addNeighbouringCells)
{
    _cellList->setNumberOfCells(7);
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    _cellList->resizeCells();
    _cellList->determineCellBoundaries(_simulationBox->getBoxDimensions());
    _cellList->addNeighbouringCells(
        settings::PotentialSettings::getCoulombRadiusCutOff()
    );

    for (const auto &cell : _cellList->getCells())
    {
        const auto &neighbourCells = cell.getNeighbourCells();
        EXPECT_EQ(neighbourCells.size(), 62);
    }

    EXPECT_EQ(
        _cellList->getNumberOfNeighbourCells(),
        linearAlgebra::Vec3Dul(2, 2, 2)
    );
}

TEST_F(TestCellList, addNeighbouringCellsRejectsAliasedPeriodicOffsets)
{
    _cellList->setNumberOfCells(2);
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    _cellList->resizeCells();
    _cellList->determineCellBoundaries(_simulationBox->getBoxDimensions());

    EXPECT_THROW_MSG(
        _cellList->addNeighbouringCells(4.0),
        customException::CellListException,
        "Invalid cell-list layout for x dimension: cell-number must be at "
        "least 2 * neighbour cells + 1 (required 3, configured 2). Decrease "
        "coulomb radius cutoff or increase cell-number."
    );
}

/**
 * @brief testing checkCoulombCutoff method
 *
 */
TEST_F(TestCellList, checkCoulombCutoff)
{
    _simulationBox->setBoxDimensions(linearAlgebra::Vec3D(50.0, 50.0, 50.0));
    _cellList->determineCellSize(_simulationBox->getBoxDimensions());
    EXPECT_NO_THROW(_cellList->checkCoulombCutoff(200.0));

    EXPECT_THROW_MSG(
        _cellList->checkCoulombCutoff(0.1),
        customException::CellListException,
        "Coulomb cutoff is smaller than half of the largest cell size."
    );
}

/* ---------- activate / deactivate / isActive ---------- */

TEST_F(TestCellList, activateDeactivateToggles_isActive)
{
    _cellList->activate();
    EXPECT_TRUE(_cellList->isActive());

    _cellList->deactivate();
    EXPECT_FALSE(_cellList->isActive());

    _cellList->activate();
    EXPECT_TRUE(_cellList->isActive());
}

TEST_F(TestCellList, resizeCellsRejectsOverflow)
{
    _cellList->setNumberOfCells(std::numeric_limits<int>::max());

    EXPECT_THROW_MSG(
        _cellList->resizeCells(),
        customException::CellListException,
        "Number of cells exceeds the supported size"
    );
}

TEST_F(TestCellList, resizeCellsRejectsZeroDimensions)
{
    _cellList->setNumberOfCells(0);

    EXPECT_THROW_MSG(
        _cellList->resizeCells(),
        customException::CellListException,
        "Number of cells must be positive"
    );
}

/* ---------- clone() copies the configured cell counts ---------- */

TEST_F(TestCellList, clone_preservesNumberOfCellsAndNeighbourCells)
{
    _cellList->setNumberOfCells(4);
    _cellList->setNumberOfNeighbourCells(2);
    _cellList->activate();

    const auto cloned = _cellList->clone();

    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->getNumberOfCells(), _cellList->getNumberOfCells());
    EXPECT_EQ(
        cloned->getNumberOfNeighbourCells(),
        _cellList->getNumberOfNeighbourCells()
    );
    EXPECT_EQ(cloned->isActive(), _cellList->isActive());
}

/**
 * @brief testing updateCellList and setup method
 *
 * TODO: think of a clever way to break this test into smaller tests
 *
 */
TEST_F(TestCellList, updateCellList)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(4.0);
    _cellList->setNumberOfCells(11);
    _cellList->resizeCells();

    EXPECT_NO_THROW(_cellList->updateCellList(*_simulationBox));
    _cellList->activate();

    auto molecule = simulationBox::Molecule();
    molecule.setNumberOfAtoms(2);

    const auto atom1 = std::make_shared<simulationBox::Atom>();
    const auto atom2 = std::make_shared<simulationBox::Atom>();

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
