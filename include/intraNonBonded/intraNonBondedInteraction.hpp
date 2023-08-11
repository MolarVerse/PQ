#ifndef _INTRA_NON_BONDED_INTERACTION_HPP_

#define _INTRA_NON_BONDED_INTERACTION_HPP_

#include "molecule.hpp"

#include <cstddef>

namespace intraNonBonded
{
    class IntraNonBondedInteraction;
}   // namespace intraNonBonded

/**
 * @class IntraNonBondedInteraction
 *
 * @brief
 */
class intraNonBonded::IntraNonBondedInteraction
{
  private:
    simulationBox::Molecule *molecule;
    size_t                   atomIndex1;
    size_t                   atomIndex2;

  public:
    explicit IntraNonBondedInteraction(simulationBox::Molecule *molecule, const size_t atomIndex1, const size_t atomIndex2)
        : molecule(molecule), atomIndex1(atomIndex1), atomIndex2(atomIndex2){};

    [[nodiscard]] simulationBox::Molecule *getMolecule() const { return molecule; }
    [[nodiscard]] size_t                   getAtomIndex1() const { return atomIndex1; }
    [[nodiscard]] size_t                   getAtomIndex2() const { return atomIndex2; }
};

#endif   // _INTRA_NON_BONDED_INTERACTION_HPP_