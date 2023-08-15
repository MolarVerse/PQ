#ifndef _Force_FIELD_HPP_

#define _Force_FIELD_HPP_

#include "angleForceField.hpp"
#include "angleType.hpp"
#include "bondForceField.hpp"
#include "bondType.hpp"
#include "defaults.hpp"
#include "dihedralForceField.hpp"
#include "dihedralType.hpp"
#include "matrix.hpp"
#include "nonCoulombPair.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace forceField
{
    class ForceField;
    enum class NonCoulombType : size_t;   // TODO: remove
    enum class MixingRule : size_t;       // TODO: remove

}   // namespace forceField

enum class forceField::NonCoulombType : size_t   // TODO: remove
{
    LJ,
    LJ_9_12,   // at the momentum just dummy for testing not implemented yet
    BUCKINGHAM,
    MORSE
};

enum class forceField::MixingRule : size_t   // TODO: remove
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
    bool           _isActivated             = false;
    bool           _isNonCoulombicActivated = false;
    NonCoulombType _nonCoulombType          = NonCoulombType::LJ;   // LJ
    MixingRule     _mixingRule              = MixingRule::NONE;     // no mixing rule

    std::vector<BondForceField>     _bonds;
    std::vector<AngleForceField>    _angles;
    std::vector<DihedralForceField> _dihedrals;
    std::vector<DihedralForceField> _improperDihedrals;

    std::vector<BondType>                                  _bondTypes;
    std::vector<AngleType>                                 _angleTypes;
    std::vector<DihedralType>                              _dihedralTypes;
    std::vector<DihedralType>                              _improperDihedralTypes;
    std::vector<std::shared_ptr<NonCoulombPair>>           _nonCoulombicPairsVector;
    linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> _nonCoulombicPairsMatrix;

  public:
    void calculateBondedInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateBondInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateAngleInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateDihedralInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateImproperDihedralInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);

    void deleteNotNeededNonCoulombicPairs(const std::vector<size_t> &);
    void determineInternalGlobalVdwTypes(const std::map<size_t, size_t> &);
    void fillDiagonalElementsOfNonCoulombicPairsMatrix(std::vector<std::shared_ptr<NonCoulombPair>> &);
    void fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

    std::vector<std::shared_ptr<NonCoulombPair>>   getSelfInteractionNonCoulombicPairs() const;
    std::optional<std::shared_ptr<NonCoulombPair>> findNonCoulombicPairByInternalTypes(size_t internalType1,
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
    void addNonCoulombicPair(std::shared_ptr<NonCoulombPair> nonCoulombicPair)
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

    void setNonCoulombType(const NonCoulombType &nonCoulombicType) { _nonCoulombType = nonCoulombicType; }

    void initNonCoulombicPairsMatrix(const size_t n)
    {
        _nonCoulombicPairsMatrix = linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>>(n);
    }

    /********************
     *                  *
     * standard getters *
     *                  *
     ********************/

    [[nodiscard]] NonCoulombType getNonCoulombType() const { return _nonCoulombType; }

    [[nodiscard]] std::vector<BondForceField>     &getBonds() { return _bonds; }
    [[nodiscard]] std::vector<AngleForceField>    &getAngles() { return _angles; }
    [[nodiscard]] std::vector<DihedralForceField> &getDihedrals() { return _dihedrals; }
    [[nodiscard]] std::vector<DihedralForceField> &getImproperDihedrals() { return _improperDihedrals; }

    [[nodiscard]] const std::vector<BondType>     &getBondTypes() const { return _bondTypes; }
    [[nodiscard]] const std::vector<AngleType>    &getAngleTypes() const { return _angleTypes; }
    [[nodiscard]] const std::vector<DihedralType> &getDihedralTypes() const { return _dihedralTypes; }
    [[nodiscard]] const std::vector<DihedralType> &getImproperDihedralTypes() const { return _improperDihedralTypes; }

    [[nodiscard]] std::vector<std::shared_ptr<NonCoulombPair>> &getNonCoulombicPairsVector() { return _nonCoulombicPairsVector; }
    [[nodiscard]] linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> &getNonCoulombicPairsMatrix()
    {
        return _nonCoulombicPairsMatrix;
    }
};

#endif   // _Force_FIELD_HPP_