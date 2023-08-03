#ifndef _Force_FIELD_HPP_

#define _Force_FIELD_HPP_

#include "angleForceField.hpp"
#include "bondForceField.hpp"
#include "defaults.hpp"
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

    double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;
    double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;

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

    /********************
     *                  *
     * standard setters *
     *                  *
     ********************/

    void setScale14Coulomb(double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
    void setScale14VanDerWaals(double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }

    /********************
     *                  *
     * standard getters *
     *                  *
     ********************/

    double getScale14Coulomb() const { return _scale14Coulomb; }
    double getScale14VanDerWaals() const { return _scale14VanDerWaals; }

    const std::vector<BondForceField>     &getBonds() const { return _bonds; }
    const std::vector<AngleForceField>    &getAngles() const { return _angles; }
    const std::vector<DihedralForceField> &getDihedrals() const { return _dihedrals; }
    const std::vector<DihedralForceField> &getImproperDihedrals() const { return _improperDihedrals; }
};

#endif   // _Force_FIELD_HPP_