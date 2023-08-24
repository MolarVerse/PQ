#ifndef _Force_FIELD_HPP_

#define _Force_FIELD_HPP_

#include "angleForceField.hpp"
#include "angleType.hpp"
#include "bondForceField.hpp"
#include "bondType.hpp"
#include "coulombPotential.hpp"
#include "defaults.hpp"
#include "dihedralForceField.hpp"
#include "dihedralType.hpp"
#include "matrix.hpp"
#include "nonCoulombPotential.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace forceField
{
    class ForceField;

}   // namespace forceField

/**
 * @class ForceField
 *
 * @brief force field object containing all force field information
 *
 */
class forceField::ForceField
{
  private:
    bool _isActivated             = false;
    bool _isNonCoulombicActivated = false;

    std::vector<BondForceField>     _bonds;
    std::vector<AngleForceField>    _angles;
    std::vector<DihedralForceField> _dihedrals;
    std::vector<DihedralForceField> _improperDihedrals;

    std::vector<BondType>     _bondTypes;
    std::vector<AngleType>    _angleTypes;
    std::vector<DihedralType> _dihedralTypes;
    std::vector<DihedralType> _improperDihedralTypes;

    std::shared_ptr<potential::NonCoulombPotential> _nonCoulombPotential;
    std::shared_ptr<potential::CoulombPotential>    _coulombPotential;

  public:
    void calculateBondedInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateBondInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateAngleInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateDihedralInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
    void calculateImproperDihedralInteractions(const simulationBox::SimulationBox &, physicalData::PhysicalData &);

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

    void clearBondTypes() { _bondTypes.clear(); }
    void clearAngleTypes() { _angleTypes.clear(); }
    void clearDihedralTypes() { _dihedralTypes.clear(); }
    void clearImproperDihedralTypes() { _improperDihedralTypes.clear(); }

    /********************
     *                  *
     * standard setters *
     *                  *
     ********************/

    void setNonCoulombPotential(const std::shared_ptr<potential::NonCoulombPotential> &nonCoulombPotential)
    {
        _nonCoulombPotential = nonCoulombPotential;
    }
    void setCoulombPotential(const std::shared_ptr<potential::CoulombPotential> &coulombPotential)
    {
        _coulombPotential = coulombPotential;
    }

    /********************
     *                  *
     * standard getters *
     *                  *
     ********************/

    [[nodiscard]] std::vector<BondForceField>     &getBonds() { return _bonds; }
    [[nodiscard]] std::vector<AngleForceField>    &getAngles() { return _angles; }
    [[nodiscard]] std::vector<DihedralForceField> &getDihedrals() { return _dihedrals; }
    [[nodiscard]] std::vector<DihedralForceField> &getImproperDihedrals() { return _improperDihedrals; }

    [[nodiscard]] const std::vector<BondType>     &getBondTypes() const { return _bondTypes; }
    [[nodiscard]] const std::vector<AngleType>    &getAngleTypes() const { return _angleTypes; }
    [[nodiscard]] const std::vector<DihedralType> &getDihedralTypes() const { return _dihedralTypes; }
    [[nodiscard]] const std::vector<DihedralType> &getImproperDihedralTypes() const { return _improperDihedralTypes; }
};

#endif   // _Force_FIELD_HPP_