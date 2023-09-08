#ifndef _CONNECTIVITY_ELEMENT_HPP_

#define _CONNECTIVITY_ELEMENT_HPP_

#include <cstddef>
#include <vector>

namespace simulationBox
{
    class Molecule;   // forward declaration
}

namespace connectivity
{

    /**
     * @class ConnectivityElement
     *
     * @brief Represents a connectivity element between n atoms.
     *
     */
    class ConnectivityElement
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

}   // namespace connectivity

#endif   // _CONNECTIVITY_ELEMENT_HPP_