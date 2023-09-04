#ifndef _BOND_HPP_

#define _BOND_HPP_

#include "connectivityElement.hpp"

#include <cstddef>
#include <vector>

namespace simulationBox
{
    class Molecule;   // forward declaration
}

namespace connectivity
{
    /**
     * @class Bond
     *
     * @brief Represents a bond between two atoms.
     *
     */
    class Bond : public ConnectivityElement
    {
      public:
        using ConnectivityElement::ConnectivityElement;
        Bond(simulationBox::Molecule *molecule1, simulationBox::Molecule *molecule2, size_t atomIndex1, size_t atomIndex2)
            : ConnectivityElement({molecule1, molecule2}, {atomIndex1, atomIndex2}){};

        /***************************
         *                         *
         * standard getter methods *
         *                         *
         ***************************/

        [[nodiscard]] simulationBox::Molecule *getMolecule1() const { return _molecules[0]; }
        [[nodiscard]] simulationBox::Molecule *getMolecule2() const { return _molecules[1]; }
        [[nodiscard]] size_t                   getAtomIndex1() const { return _atomIndices[0]; }
        [[nodiscard]] size_t                   getAtomIndex2() const { return _atomIndices[1]; }
    };

}   // namespace connectivity

#endif   // _BOND_HPP_