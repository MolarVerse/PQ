#ifndef _INTRA_NON_BONDED_MAP_HPP_

#define _INTRA_NON_BONDED_MAP_HPP_

#include "intraNonBondedContainer.hpp"
#include "molecule.hpp"

#include <cstddef>

namespace intraNonBonded
{
    class IntraNonBondedMap;
}   // namespace intraNonBonded

/**
 * @class IntraNonBondedInteraction
 *
 * @brief
 */
class intraNonBonded::IntraNonBondedMap
{
  private:
    simulationBox::Molecule *molecule;
    IntraNonBondedContainer *_intraNonBondedType;

  public:
    explicit IntraNonBondedMap(simulationBox::Molecule *molecule, IntraNonBondedContainer *intraNonBondedType)
        : molecule(molecule), _intraNonBondedType(intraNonBondedType)
    {
    }

    [[nodiscard]] IntraNonBondedContainer *getIntraNonBondedType() const { return _intraNonBondedType; }
    [[nodiscard]] simulationBox::Molecule *getMolecule() const { return molecule; }
};

#endif   // _INTRA_NON_BONDED_MAP_HPP_