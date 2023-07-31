#ifndef _BOND_HPP_

#define _BOND_HPP_

#include "connectivityElement.hpp"

namespace connectivity
{
    class Bond;
}   // namespace connectivity

/**
 * @class Bond
 *
 * @brief Represents a bond between two atoms.
 *
 */
class connectivity::Bond : public connectivity::ConnectivityElement
{
  public:
    using connectivity::ConnectivityElement::ConnectivityElement;
    Bond(simulationBox::Molecule *molecule1, simulationBox::Molecule *molecule2, size_t atomIndex1, size_t atomIndex2)
        : connectivity::ConnectivityElement({molecule1, molecule2}, {atomIndex1, atomIndex2}){};

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    simulationBox::Molecule *getMolecule1() const { return _molecules[0]; }
    simulationBox::Molecule *getMolecule2() const { return _molecules[1]; }
    size_t                   getAtomIndex1() const { return _atomIndices[0]; }
    size_t                   getAtomIndex2() const { return _atomIndices[1]; }
};

#endif   // _BOND_HPP_