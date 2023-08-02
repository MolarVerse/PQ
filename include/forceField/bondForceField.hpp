#ifndef _BOND_FORCE_FIELD_HPP_

#define _BOND_FORCE_FIELD_HPP_

#include "bond.hpp"
#include "molecule.hpp"

#include <cstddef>

namespace forceField
{
    class BondForceField;
}

/**
 * @class BondForceField inherits from Bond
 *
 * @brief force field object for single bond length
 *
 */
class forceField::BondForceField : public connectivity::Bond
{
  private:
    size_t _type;

  public:
    BondForceField(
        simulationBox::Molecule *molecule1, simulationBox::Molecule *molecule2, size_t atomIndex1, size_t atomIndex2, size_t type)
        : connectivity::Bond(molecule1, molecule2, atomIndex1, atomIndex2), _type(type){};

    size_t getType() const { return _type; }
};

#endif   // _BOND_FORCE_FIELD_HPP_