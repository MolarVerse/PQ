#ifndef _Force_FIELD_HPP_

#define _Force_FIELD_HPP_

#include "angleForceField.hpp"
#include "bondForceField.hpp"
#include "dihedralForceField.hpp"

#include <vector>

namespace forceField
{
    class ForceField;
}

/**
 * @class ForceField
 *
 * @brief force field object containing all force field information
 *
 */
class forceField::ForceField
{
  private:
    bool _isActivated = false;

    std::vector<BondForceField>     _bonds;
    std::vector<AngleForceField>    _angles;
    std::vector<DihedralForceField> _dihedrals;
    std::vector<DihedralForceField> _improperDihedrals;

  public:
    void addBond(const BondForceField &bond) { _bonds.push_back(bond); }
    void addAngle(const AngleForceField &angle) { _angles.push_back(angle); }
    void addDihedral(const DihedralForceField &dihedral) { _dihedrals.push_back(dihedral); }
    void addImproperDihedral(const DihedralForceField &improperDihedral) { _improperDihedrals.push_back(improperDihedral); }

    void activate() { _isActivated = true; }
    void deactivate() { _isActivated = false; }
    bool isActivated() const { return _isActivated; }

    const std::vector<BondForceField>     &getBonds() const { return _bonds; }
    const std::vector<AngleForceField>    &getAngles() const { return _angles; }
    const std::vector<DihedralForceField> &getDihedrals() const { return _dihedrals; }
    const std::vector<DihedralForceField> &getImproperDihedrals() const { return _improperDihedrals; }
};

#endif   // _Force_FIELD_HPP_