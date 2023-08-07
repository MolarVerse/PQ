#ifndef _CONNECTIVITY_ELEMENT_HPP_

#define _CONNECTIVITY_ELEMENT_HPP_

#include "molecule.hpp"

#include <cstddef>
#include <vector>

namespace connectivity
{
    class ConnectivityElement;
}   // namespace connectivity

/**
 * @class ConnectivityElement
 *
 * @brief Represents a connectivity element between n atoms.
 *
 */
class connectivity::ConnectivityElement
{
  protected:
    std::vector<simulationBox::Molecule *> _molecules;
    std::vector<size_t>                    _atomIndices;

  public:
    ConnectivityElement(const std::vector<simulationBox::Molecule *> &molecules, const std::vector<size_t> &atomIndices)
        : _molecules(molecules), _atomIndices(atomIndices){};

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    [[nodiscard]] std::vector<simulationBox::Molecule *> getMolecules() const { return _molecules; }
    [[nodiscard]] std::vector<size_t>                    getAtomIndices() const { return _atomIndices; }
};

#endif   // _CONNECTIVITY_ELEMENT_HPP_