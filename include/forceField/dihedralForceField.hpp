#ifndef _DIHEDRAL_FORCE_FIELD_HPP_

#define _DIHEDRAL_FORCE_FIELD_HPP_

#include "dihedral.hpp"
#include "molecule.hpp"

#include <cstddef>
#include <vector>

namespace forceField
{
    class DihedralForceField;
}   // namespace forceField

/**
 * @class DihedralForceField
 *
 * @brief Represents a dihedral between four atoms.
 *
 */
class forceField::DihedralForceField : public connectivity::Dihedral
{
  private:
    size_t _type;

  public:
    DihedralForceField(const std::vector<simulationBox::Molecule *> &molecules,
                       const std::vector<size_t>                    &atomIndices,
                       size_t                                        type)
        : connectivity::Dihedral(molecules, atomIndices), _type(type){};

    size_t getType() const { return _type; }
};

#endif   // _DIHEDRAL_FORCE_FIELD_HPP_
