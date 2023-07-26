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

    size_t _atomindex1;
    size_t _atomindex2;

    double _bondLength;

  public:
    BondConstraint(const simulationBox::Molecule *molecule1,
                   const simulationBox::Molecule *molecule2,
                   size_t                         atomindex1,
                   size_t                         atomindex2,
                   double                         bondLength)
        : _molecule1(molecule1), _molecule2(molecule2), _atomindex1(atomindex1), _atomindex2(atomindex2),
          _bondLength(bondLength){};
};

#endif   // _BOND_CONSTRAINT_HPP_