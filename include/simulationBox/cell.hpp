#ifndef _CELL_HPP_

#define _CELL_HPP_

#include "vector3d.hpp"   // for Vec3D, Vec3Dul

#include <cstddef>   // for size_t
#include <vector>    // for vector

namespace simulationBox
{
    class Molecule;   // forward declaration

    /**
     * @class Cell
     *
     * @brief Cell is a class for a single cell in the cellList
     *
     */
    class Cell
    {
      private:
        std::vector<Molecule *>          _molecules;
        std::vector<std::vector<size_t>> _atomIndices;
        std::vector<Cell *>              _neighbourCells;

        linearAlgebra::Vec3D   _lowerBoundary = {0, 0, 0};
        linearAlgebra::Vec3D   _upperBoundary = {0, 0, 0};
        linearAlgebra::Vec3Dul _cellIndex     = {0, 0, 0};

      public:
        void clearMolecules() { _molecules.clear(); }
        void clearAtomIndices() { _atomIndices.clear(); }

        void addMolecule(Molecule &molecule) { _molecules.push_back(&molecule); }
        void addMolecule(Molecule *molecule) { _molecules.push_back(molecule); }
        void addNeighbourCell(Cell *cell) { _neighbourCells.push_back(cell); }
        void addAtomIndices(const std::vector<size_t> &atomIndices) { _atomIndices.push_back(atomIndices); }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t                        getNumberOfMolecules() const { return _molecules.size(); }
        [[nodiscard]] size_t                        getNumberOfNeighbourCells() const { return _neighbourCells.size(); }
        [[nodiscard]] const linearAlgebra::Vec3D   &getLowerBoundary() const { return _lowerBoundary; }
        [[nodiscard]] const linearAlgebra::Vec3D   &getUpperBoundary() const { return _upperBoundary; }
        [[nodiscard]] const linearAlgebra::Vec3Dul &getCellIndex() const { return _cellIndex; }

        [[nodiscard]] Molecule               *getMolecule(const size_t index) const { return _molecules[index]; }
        [[nodiscard]] std::vector<Molecule *> getMolecules() const { return _molecules; }

        [[nodiscard]] Cell               *getNeighbourCell(const size_t index) const { return _neighbourCells[index]; }
        [[nodiscard]] std::vector<Cell *> getNeighbourCells() const { return _neighbourCells; }

        [[nodiscard]] const std::vector<size_t> &getAtomIndices(const size_t index) const { return _atomIndices[index]; }

        /***************************
         * standard setter methods *
         ***************************/

        void setLowerBoundary(const linearAlgebra::Vec3D &lowerBoundary) { _lowerBoundary = lowerBoundary; }
        void setUpperBoundary(const linearAlgebra::Vec3D &upperBoundary) { _upperBoundary = upperBoundary; }
        void setCellIndex(const linearAlgebra::Vec3Dul &cellIndex) { _cellIndex = cellIndex; }
    };

}   // namespace simulationBox

#endif   // _CELL_HPP_