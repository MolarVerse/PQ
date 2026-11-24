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

#ifndef MOLSYS__CELLLIST_HPP
#define MOLSYS__CELLLIST_HPP

#include <cstddef>
#include <vector>

#include "cell.hpp"
#include "defaults.hpp"
#include "unit_types.hpp"
#include "vector3d.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration

    /**
     * @class CellList
     *
     * @brief CellList is a class for cell list
     *
     */
    class CellList
    {
       private:
        std::vector<Cell> _cells;

        Pos3D           _cellSize;
        linalg::Vec3Dul _nNeighbourCells{0, 0, 0};
        linalg::Vec3Dul _nCells{defaults::NUMBER_OF_CELLS_DEFAULT};

        std::optional<Pos3D>   _oldBoxDimensions = std::nullopt;
        std::optional<Angle3D> _oldBoxAngles     = std::nullopt;

       public:
        [[nodiscard]] std::shared_ptr<CellList> clone() const;

        void setup(
            const SimulationBox &simulationBox,
            const Distance      &coulombRadiusCutoff
        );
        void updateCellList(
            SimulationBox         &simulationBox,
            const Distance        &coulombRadiusCutoff,
            const GeneralSettings &generalSettings
        );

        void determineCellSize(const Pos3D &box);
        void determineCellBoundaries(const Pos3D &box);
        void checkCoulombCutoff(const Pos &coulombCutoff) const;

        void addNeighbouringCells(const Pos &coulombCutoff);
        void addNeighbouringCellPointers(Cell &);
        void addMoleculesToCells(SimulationBox &simulationBox);
        void assignMoleculeHybridZoneIndices();
        void assignWaterMoleculeIndices(
            const SimulationBox &,
            const WaterModelSettings &
        );

        [[nodiscard]] size_t getCellIndex(
            const linalg::Vec3Dul &cellIndices
        ) const;
        [[nodiscard]] linalg::Vec3Dul getCellIndexOfAtom(
            const Pos3D &,
            const Pos3D &
        ) const;

        void resizeCells();
        void addCell(const Cell &cell);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] linalg::Vec3Dul getNumberOfCells() const;
        [[nodiscard]] linalg::Vec3Dul getNumberOfNeighbourCells() const;
        [[nodiscard]] const Pos3D    &getCellSize() const;
        [[nodiscard]] const std::vector<Cell> &getCells() const;
        [[nodiscard]] Cell                    &getCell(size_t index);

        /***************************
         * standard setter methods *
         ***************************/

        void setNumberOfCells(size_t nCells);
        void setNumberOfNeighbourCells(size_t nCells);

       private:
        [[nodiscard]]
        bool _hasBoxSizeChanged(
            const molsys::SimulationBox &simulationBox
        ) const;
    };

}   // namespace molsys

#endif   // MOLSYS__CELLLIST_HPP
