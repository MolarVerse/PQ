#ifndef _BOND_CONSTRAINT_HPP_

#define _BOND_CONSTRAINT_HPP_

#include "molecule.hpp"

namespace constraints
{
    class BondConstraint;
}

/**
 * @class BondConstraint
 *
 * @brief constraint for bond lengths
 *
 */
class constraints::BondConstraint
{
  private:
    const simulationBox::Molecule *_molecule1;
    const simulationBox::Molecule *_molecule2;

    size_t _atomIndex1;
    size_t _atomIndex2;

    double _bondLength;

  public:
    BondConstraint(const simulationBox::Molecule *molecule1,
                   const simulationBox::Molecule *molecule2,
                   size_t                         atomIndex1,
                   size_t                         atomIndex2,
                   double                         bondLength)
        : _molecule1(molecule1), _molecule2(molecule2), _atomIndex1(atomIndex1), _atomIndex2(atomIndex2),
          _bondLength(bondLength){};

    const simulationBox::Molecule *getMolecule1() const { return _molecule1; }
    const simulationBox::Molecule *getMolecule2() const { return _molecule2; }
    size_t                         getAtomIndex1() const { return _atomIndex1; }
    size_t                         getAtomIndex2() const { return _atomIndex2; }
    double                         getBondLength() const { return _bondLength; }
};

#endif   // _BOND_CONSTRAINT_HPP_