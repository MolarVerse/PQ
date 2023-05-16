#ifndef _CELL_H_

#define _CELL_H_

#include <vector>

#include "molecule.hpp"

/**
 * @class Cell
 *
 * @brief Cell is a class for cell
 *
 */
class Cell
{
private:
    std::vector<Molecule *> _molecules;
    std::vector<Cell> _neighbourCells;

public:
    void addMolecule(Molecule *molecule) { _molecules.push_back(molecule); }
    std::vector<Molecule *> getMolecules() const { return _molecules; }
};

#endif // _CELL_H_