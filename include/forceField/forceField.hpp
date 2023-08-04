#ifndef _Force_FIELD_HPP_

#define _Force_FIELD_HPP_

#include "angleForceField.hpp"
#include "angleType.hpp"
#include "bondForceField.hpp"
#include "bondType.hpp"
#include "defaults.hpp"
#include "dihedralForceField.hpp"
#include "dihedralType.hpp"
#include "nonCoulombicPair.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace forceField
{
    class ForceField;
    enum NonCoulombicType : size_t;

}   // namespace forceField

enum forceField::NonCoulombicType : size_t
{
    LJ,
    LJ_9_12,   // at the momentum just dummy for testing not implemented yet
    BUCKINGHAM,
    MORSE
};

/**
 * @class ForceField
 *
 * @brief force field object containing all force field information
 *
 */
class forceField::ForceField
{
  private:
    bool             _isActivated      = false;
    NonCoulombicType _nonCoulombicType = NonCoulombicType::LJ;   // LJ

    double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;
    double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;

    std::vector<BondForceField>     _bonds;
    std::vector<AngleForceField>    _angles;
    std::vector<DihedralForceField> _dihedrals;
    std::vector<DihedralForceField> _improperDihedrals;

    std::vector<BondType>                          _bondTypes;
    std::vector<AngleType>                         _angleTypes;
    std::vector<DihedralType>                      _dihedralTypes;
    std::vector<DihedralType>                      _improperDihedralTypes;
    std::vector<std::unique_ptr<NonCoulombicPair>> _nonCoulombicPairs;

  public:
    const BondType     &findBondTypeById(size_t id) const;
    const AngleType    &findAngleTypeById(size_t id) const;
    const DihedralType &findDihedralTypeById(size_t id) const;
    const DihedralType &findImproperDihedralTypeById(size_t id) const;

    void activate() { _isActivated = true; }
    void deactivate() { _isActivated = false; }
    bool isActivated() const { return _isActivated; }

    void addBond(const BondForceField &bond) { _bonds.push_back(bond); }
    void addAngle(const AngleForceField &angle) { _angles.push_back(angle); }
    void addDihedral(const DihedralForceField &dihedral) { _dihedrals.push_back(dihedral); }
    void addImproperDihedral(const DihedralForceField &improperDihedral) { _improperDihedrals.push_back(improperDihedral); }

    void addBondType(const BondType &bondType) { _bondTypes.push_back(bondType); }
    void addAngleType(const AngleType &angleType) { _angleTypes.push_back(angleType); }
    void addDihedralType(const DihedralType &dihedralType) { _dihedralTypes.push_back(dihedralType); }
    void addImproperDihedralType(const DihedralType &improperDihedralType)
    {
        _improperDihedralTypes.push_back(improperDihedralType);
    }
    void addNonCoulombicPair(std::unique_ptr<NonCoulombicPair> nonCoulombicPair)
    {
        _nonCoulombicPairs.push_back(std::move(nonCoulombicPair));
    }

    void clearBondTypes() { _bondTypes.clear(); }
    void clearAngleTypes() { _angleTypes.clear(); }
    void clearDihedralTypes() { _dihedralTypes.clear(); }
    void clearImproperDihedralTypes() { _improperDihedralTypes.clear(); }
    void clearNonCoulombicPairs() { _nonCoulombicPairs.clear(); }

    /********************
     *                  *
     * standard setters *
     *                  *
     ********************/

    void setScale14Coulomb(double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
    void setScale14VanDerWaals(double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }

    void setNonCoulombicType(const NonCoulombicType &nonCoulombicType) { _nonCoulombicType = nonCoulombicType; }

    /********************
     *                  *
     * standard getters *
     *                  *
     ********************/

    double getScale14Coulomb() const { return _scale14Coulomb; }
    double getScale14VanDerWaals() const { return _scale14VanDerWaals; }

    NonCoulombicType getNonCoulombicType() const { return _nonCoulombicType; }

    const std::vector<BondForceField>     &getBonds() const { return _bonds; }
    const std::vector<AngleForceField>    &getAngles() const { return _angles; }
    const std::vector<DihedralForceField> &getDihedrals() const { return _dihedrals; }
    const std::vector<DihedralForceField> &getImproperDihedrals() const { return _improperDihedrals; }

    const std::vector<BondType>     &getBondTypes() const { return _bondTypes; }
    const std::vector<AngleType>    &getAngleTypes() const { return _angleTypes; }
    const std::vector<DihedralType> &getDihedralTypes() const { return _dihedralTypes; }
    const std::vector<DihedralType> &getImproperDihedralTypes() const { return _improperDihedralTypes; }

    std::vector<std::unique_ptr<NonCoulombicPair>> &getNonCoulombicPairs() { return _nonCoulombicPairs; }
};

#endif   // _Force_FIELD_HPP_