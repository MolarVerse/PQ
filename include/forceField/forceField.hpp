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
#include "staticMatrix.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace forceField
{
    class ForceField;
    enum class NonCoulombicType : size_t;
    enum class MixingRule : size_t;

}   // namespace forceField

enum class forceField::NonCoulombicType : size_t
{
    LJ,
    LJ_9_12,   // at the momentum just dummy for testing not implemented yet
    BUCKINGHAM,
    MORSE
};

enum class forceField::MixingRule : size_t
{
    NONE
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
    bool             _isActivated             = false;
    bool             _isNonCoulombicActivated = false;
    NonCoulombicType _nonCoulombicType        = NonCoulombicType::LJ;   // LJ
    MixingRule       _mixingRule              = MixingRule::NONE;       // no mixing rule

    double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;
    double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;

    std::vector<BondForceField>     _bonds;
    std::vector<AngleForceField>    _angles;
    std::vector<DihedralForceField> _dihedrals;
    std::vector<DihedralForceField> _improperDihedrals;

    std::vector<BondType>                                    _bondTypes;
    std::vector<AngleType>                                   _angleTypes;
    std::vector<DihedralType>                                _dihedralTypes;
    std::vector<DihedralType>                                _improperDihedralTypes;
    std::vector<std::unique_ptr<NonCoulombicPair>>           _nonCoulombicPairsVector;
    linearAlgebra::Matrix<std::unique_ptr<NonCoulombicPair>> _nonCoulombicPairsMatrix;

  public:
    void deleteNotNeededNonCoulombicPairs(const std::vector<size_t> &);
    void determineInternalGlobalVdwTypes(const std::map<size_t, size_t> &);
    void fillDiagonalElementsOfNonCoulombicPairsMatrix(std::vector<std::unique_ptr<NonCoulombicPair>> &);
    void fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

    std::vector<std::unique_ptr<NonCoulombicPair>>   getSelfInteractionNonCoulombicPairs() const;
    std::optional<std::unique_ptr<NonCoulombicPair>> findNonCoulombicPairByInternalTypes(size_t internalType1,
                                                                                         size_t internalType2) const;

    const BondType     &findBondTypeById(size_t id) const;
    const AngleType    &findAngleTypeById(size_t id) const;
    const DihedralType &findDihedralTypeById(size_t id) const;
    const DihedralType &findImproperDihedralTypeById(size_t id) const;

    void               activate() { _isActivated = true; }
    void               deactivate() { _isActivated = false; }
    [[nodiscard]] bool isActivated() const { return _isActivated; }

    void               activateNonCoulombic() { _isNonCoulombicActivated = true; }
    void               deactivateNonCoulombic() { _isNonCoulombicActivated = false; }
    [[nodiscard]] bool isNonCoulombicActivated() const { return _isNonCoulombicActivated; }

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
        _nonCoulombicPairsVector.push_back(std::move(nonCoulombicPair));
    }

    void clearBondTypes() { _bondTypes.clear(); }
    void clearAngleTypes() { _angleTypes.clear(); }
    void clearDihedralTypes() { _dihedralTypes.clear(); }
    void clearImproperDihedralTypes() { _improperDihedralTypes.clear(); }
    void clearNonCoulombicPairs() { _nonCoulombicPairsVector.clear(); }

    /********************
     *                  *
     * standard setters *
     *                  *
     ********************/

    void setScale14Coulomb(const double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
    void setScale14VanDerWaals(const double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }

    void setNonCoulombicType(const NonCoulombicType &nonCoulombicType) { _nonCoulombicType = nonCoulombicType; }

    void initNonCoulombicPairsMatrix(const size_t n)
    {
        _nonCoulombicPairsMatrix = linearAlgebra::Matrix<std::unique_ptr<NonCoulombicPair>>(n);
    }

    /********************
     *                  *
     * standard getters *
     *                  *
     ********************/

    [[nodiscard]] double getScale14Coulomb() const { return _scale14Coulomb; }
    [[nodiscard]] double getScale14VanDerWaals() const { return _scale14VanDerWaals; }

    [[nodiscard]] NonCoulombicType getNonCoulombicType() const { return _nonCoulombicType; }

    [[nodiscard]] std::vector<BondForceField>     &getBonds() { return _bonds; }
    [[nodiscard]] std::vector<AngleForceField>    &getAngles() { return _angles; }
    [[nodiscard]] std::vector<DihedralForceField> &getDihedrals() { return _dihedrals; }
    [[nodiscard]] std::vector<DihedralForceField> &getImproperDihedrals() { return _improperDihedrals; }

    [[nodiscard]] const std::vector<BondType>     &getBondTypes() const { return _bondTypes; }
    [[nodiscard]] const std::vector<AngleType>    &getAngleTypes() const { return _angleTypes; }
    [[nodiscard]] const std::vector<DihedralType> &getDihedralTypes() const { return _dihedralTypes; }
    [[nodiscard]] const std::vector<DihedralType> &getImproperDihedralTypes() const { return _improperDihedralTypes; }

    [[nodiscard]] std::vector<std::unique_ptr<NonCoulombicPair>> &getNonCoulombicPairsVector()
    {
        return _nonCoulombicPairsVector;
    }
};

#endif   // _Force_FIELD_HPP_